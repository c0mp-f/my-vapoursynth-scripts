import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import lvsfunc as lvf
import G41Fun as G41
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split
from finedehalo import fine_dehalo
from adptvgrnMod import adptvgrnMod
from zzfunc.std import LevelsM
from acsuite import eztrim
from vsdpir import DPIR

core = vs.core
filepath = "/mnt/HDD1/lupin/"

# OP/ED
denoise_exclude = [(1416, 3500), (30449, 32127)]

aa_include = [(5262, 5893), (6579, 6612), (7225, 8990), 
            (9111, 9356), (14893, 15036), (18614, 18655), (22523, 22690), 
            (30377, 30448)]

stronger_deband_ranges = [(5479, 6620), (7039, 7098), (7615, 7710), 
        (8566, 8792), (10158, 10217), (12037, 13124), (13389, 13448), 
        (15109, 15204), (18482, 19513), (19680, 19706), (28243, 29148), (32603, 32656)]

dehalo_chroma_range = [(594, 1415), (3501, 5003), (5702, 5947), (7225, 8792), (10158, 11898), (12031, 13500), (15297, None)]

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

    mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
    return mask

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep01/00002.m2ts")
src = core.std.Crop(gt, top=2)[:32848]
src = depth(src, 16)

# denoise/deblock
uv_denoise = core.knlm.KNLMeansCL(src, d=3, a=4, h=0.5, channels='UV', device_type='gpu')
deblock = haf.SMDegrain(uv_denoise, thSAD=200)
denoise = lvf.rfs(deblock, uv_denoise, denoise_exclude)

# AA
transpose = lvf.aa.transpose_aa(denoise, eedi3=True)
sraa = lvf.sraa(denoise, rfactor=1.5)

clamp = lvf.aa.clamp_aa(denoise, weak=transpose, strong=sraa, strength=10)
aa = lvf.rfs(denoise, clamp, aa_include)

# dehalo
fine_dehalo_args = {"darkstr":0, "brightstr":0.6, "rx":3, "thlimi":30, "thlima":128, "thmi":35, "thma": 100, "lowsens":50, "highsens":50}

dehalo_chroma = join([
	plane(aa, 0),
	fine_dehalo(plane(aa, 1), edgemask=vsmask.edge.Sobel(), **fine_dehalo_args),
	fine_dehalo(plane(aa, 2), edgemask=vsmask.edge.Sobel(), **fine_dehalo_args)
])

dehalo = lvf.rfs(aa, dehalo_chroma, dehalo_chroma_range)

# deband
deband = debandshit.dumb3kdb(dehalo, radius=19, threshold=[32, 32, 32], grain=0)

detail_mask = lvf.mask.detail_mask(denoise, rad=4, brz_a=0.01, brz_b=0.02)
texture = texture_mask(denoise, radius=3)

deband = lvf.rfs(deband, denoise, denoise_exclude)

strong_deband = debandshit.dumb3kdb(dehalo, radius=25, threshold=[60, 44, 44])
deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, dehalo, detail_mask)
deband = core.std.MaskedMerge(deband, dehalo, texture)

cope = texture_mask(denoise, radius=10)
sceneflt_deband = core.std.MaskedMerge(deband, dehalo, cope)
deband = lvf.rfs(deband, sceneflt_deband, [(8712, 8792), (18614, 18655)])

grain = adptvgrnMod(deband, strength=0.35, cstrength=0.15, size=1.25, sharp=90, luma_scaling=24)

# ED scenefilter
# Tons of ringing, on text most noticably :|

ed = dehalo[30449:32127]

# make the same length and resolution
nced = core.std.Crop(nced, top=2)[:1678]

credit_mask = get_y(core.std.MakeDiff(ed, nced))
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 4)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

dering = to_yuvps(DPIR(to_rgbs(ed), strength=25, task="deblock"))
dering = adptvgrnMod(dering, strength=0.9, luma_scaling=90, seed=1)

ending = core.std.MaskedMerge(ed, dering, rings)

noise_diff = core.bilateralgpu.Bilateral(dering, sigma_spatial=2, sigma_color=0.25, radius=2)
diff = core.std.MakeDiff(ending, noise_diff)

ed_deband = debandshit.dumb3kdb(noise_diff, radius=25, threshold=[64, 32, 32])
ed_grain = adptvgrnMod(ed_deband, strength=0.9, cstrength=0.1, size=1.5, sharp=125, luma_scaling=90, seed=1)
ed_deband = core.std.MergeDiff(ed_grain, diff)

ending = insert_clip(grain, ed_deband, 30449)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep01/00002.opus")
final.set_output()
#vspipe --filter-time Lupin01.vpy - -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --no-open-gop --no-cutree --rd 3 --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 16 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2.1 --ipratio 1.25 --pbratio 1.15 --zones 31,239,b=0.85/1416,3500,b=0.9/3633,3707,q=7/30449,32127,b=0.9 - -o /mnt/HDD1/lupin/ep01/final.hevc
