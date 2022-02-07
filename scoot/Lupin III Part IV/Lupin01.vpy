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

# false positives outweigh actual haloing
dehalo_exclude = [(32127, 32262)]

aa_include = [(None, 3500), (5124, 5893), (6579, 6612), (7225, 8990), 
            (9111, 10157), (13389, 13448), (14118, 14261), (14893, 15036),
            (17734, 18112), (18614, 21245), (22523, 22690), (25319, 31993), (32128, None)]

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
uv_denoise = core.knlm.KNLMeansCL(src, d=3, a=4, h=0.75, channels='UV', device_type='gpu')
deblock = to_yuvps(DPIR(to_rgbs(src), task="deblock", strength=10)) 
denoise = lvf.rfs(deblock, uv_denoise, denoise_exclude)

# AA
transpose = lvf.aa.transpose_aa(denoise, eedi3=True)
sraa = lvf.sraa(denoise, rfactor=1.4)

clamp = lvf.aa.clamp_aa(denoise, weak=transpose, strong=sraa, strength=2.5)
aa = lvf.rfs(denoise, clamp, aa_include)

# dehalo
dehalo = fine_dehalo(aa, rx=3, darkstr=0, brightstr=0.8, thlimi=50, thlima=100)
dehalo = lvf.rfs(dehalo, aa, dehalo_exclude)

sobel = vsmask.edge.ExSobel().get_mask(get_y(denoise))
sobel_brz = core.std.Binarize(sobel, 45<<8)
sobel_big = core.std.Maximum(sobel_brz)
sobel_small = iterate(sobel_big, core.std.Minimum, 2)
difference = core.std.Expr([sobel_brz, sobel_big], "x y - abs")

dehalo_args = {"rx":3, "brightstr":0.8, "ss":1.5, "maskpush":120, "maskpull":18, "lowsens":50, "highsens":50}

dehalo_chroma = join([
	plane(dehalo, 0),
	G41.MaskedDHA(plane(dehalo, 1), darkstr=0, **dehalo_args),
	G41.MaskedDHA(plane(dehalo, 2), darkstr=0, **dehalo_args)
])

dehalo_chroma = lvf.rfs(dehalo, dehalo_chroma, dehalo_chroma_range)

dehalo = lvf.rfs(dehalo_chroma, denoise, dehalo_exclude)

# deband
deband = debandshit.dumb3kdb(dehalo, radius=19, threshold=[32, 32, 32], grain=0)

detail_mask = lvf.mask.detail_mask(src, rad=3, brz_a=0.012, brz_b=0.03)
texture = texture_mask(denoise, radius=2)

deband = lvf.rfs(deband, denoise, denoise_exclude)

strong_deband = debandshit.dumb3kdb(dehalo, radius=25, threshold=[60, 44, 44])
deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, dehalo, detail_mask)
deband = core.std.MaskedMerge(deband, dehalo, texture)
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

blur = core.bilateralgpu.Bilateral(ed, 2, 0.25, radius=2)

ending = core.std.MaskedMerge(ed, blur, rings)

noise_diff = core.std.BoxBlur(ending, hradius=2, vradius=2)
diff = core.std.MakeDiff(ending, noise_diff)

ed_deband = debandshit.dumb3kdb(noise_diff, radius=25, threshold=[32, 16, 16])
ed_grain = adptvgrnMod(ed_deband, strength=0.9, cstrength=0.1, size=1.5, sharp=75, luma_scaling=100)
ed_deband = core.std.MergeDiff(ed_grain, diff)

ending = insert_clip(grain, ed_deband, 30449)

final = depth(ending, 10)

eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep01/00002.opus")
final.set_output()

#vspipe - --filter-time Lupin01.vpy -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --no-rect --no-amp --no-open-gop --no-cutree --rc-lookahead 120 --ref 6 --bframes 16 --rd 3 --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs 0 --crqpoffs 0 --qcomp 0.75 --crf 15.75 --no-sao --aq-mode 4 --aq-strength 0.75 --psy-rd 1.75 --psy-rdoq 1.75 --ipratio 1.35 --pbratio 1.25 --deblock 0:0 - --zones 3633,3707,q=7/8374,8990,q=9/28243,29148,q=9/30308,30406,q=10 -o /mnt/HDD1/lupin/ep01/final.hevc
