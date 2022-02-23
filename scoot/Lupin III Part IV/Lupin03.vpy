import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import lvsfunc as lvf
import G41Fun as G41
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split
from adptvgrnMod import adptvgrnMod
from zzfunc.std import LevelsM
from acsuite import eztrim
from vsdpir import DPIR

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

aa_ranges = [(21791, 21973)]
stronger_deband_ranges = [(4572, 4621), (9709, 9846), (10860, 11912), 
					(12185, 12593), (16515, 16555), (17889, 17928), 
					(18351, 19463), (24585, 25617), (25977, 26624), 
					(29769, 29942), (29943, 30166)]

placebo_deband_ranges = [(650, 2373), (4261, 6218), (7656, 9018), (9967, 10030), (16785, 16818), (20051, 20146)]
denoise_exclude = [(2374, 4460), (12097, 12183), (30449, 32127)]
crepair_range = [(4741, 8544), (8599, 9127), (9444, 9708), 
			(9847, 10188), (10668, 10715), (10860, 12097), 
			(14566, 16422), (16554, 25616), (25977, 26624), 
			(26885, 30400), (32671, None)]

unfuck_harder = [(13906, 14380), (15284, 15370), (15443, 15682), (17536, 17912), (19485, 21685), (22034, 26624), (26885, 28526)]
deblock_necessary = [(9847, 9966), (4815, 5044), (11091, 11354), (11481, 11588), (29953, 30322), (30239, 30442)]


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

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep03/00004.m2ts")
src = core.std.Crop(gt, top=2)[:32848]
src = depth(src, 16)

# denoise/deblock
uv_denoise = lvf.rfs(src, core.knlm.KNLMeansCL(src, d=3, a=4, h=0.75, channels='UV', device_type='gpu'), denoise_exclude[0])

deblock = core.dfttest.DFTTest(src, sigma=6)

denoise = haf.SMDegrain(deblock, tr=6, thSAD=200, contrasharp=True)
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)
denoise = lvf.rfs(denoise, deblock, deblock_necessary)

pre = core.dfttest.DFTTest(src, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.075, brz_b=0.01)
texture = texture_mask(pre, radius=4)

denoise = core.std.MaskedMerge(denoise, src, detail_mask)

crepair = to_yuvps(core.w2xc.Waifu2x(to_rgbs(denoise), scale=1, noise=1))
crepair = vdf.merge_chroma(denoise, crepair)
crepair = lvf.rfs(denoise, crepair, crepair_range)

crepair_harder = to_yuvps(core.w2xc.Waifu2x(to_rgbs(denoise), scale=1, noise=2))
crepair_harder = vdf.merge_chroma(denoise, crepair)
crepair = lvf.rfs(crepair, crepair_harder, unfuck_harder)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)
sraa = lvf.sraa(crepair, rfactor=1.4)
aa = lvf.aa.clamp_aa(src=crepair, weak=transpose, strong=sraa, strength=3)
aa = lvf.rfs(crepair, aa, aa_ranges)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[40, 32, 32], grain=0)

deband = lvf.rfs(deband, denoise, denoise_exclude)
stronger_deband = debandshit.dumb3kdb(aa, radius=26, threshold=[60, 60, 60])
deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges) 

placebo_deband = core.placebo.Deband(aa, radius=18, iterations=1, threshold=6, grain=0, planes=1, dither=0)
placebo_deband = core.placebo.Deband(placebo_deband, radius=18, iterations=1, threshold=7, grain=0, planes=2|4, dither=0)
deband = lvf.rfs(deband, placebo_deband, placebo_deband_ranges)
# else ugly "half deband half detail"
deband = lvf.rfs(deband, debandshit.dumb3kdb(aa, radius=19, threshold=[32, 12, 12]), [(4961, 5044), (9847, 9966)])

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.24, cstrength=0.1, size=1.5, sharp=69, luma_scaling=24, seed=1)
grain = adptvgrnMod(grain, strength=0.15, cstrength=0.1, size=1.15, sharp=55, luma_scaling=36, seed=2)

# ED scenefilter

ed = aa[30451:32129]

dering = to_yuvps(DPIR(to_rgbs(ed), strength=50, task="deblock"))

nced = core.std.Crop(nced, top=2)[:1678]

credit_mask = get_y(core.std.MakeDiff(ed, nced))
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 5)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ending = core.std.MaskedMerge(ed, dering, rings)

noise_diff = core.std.BoxBlur(ending, hradius=2, vradius=2)
diff = core.std.MakeDiff(ending, dering)

ed_deband = debandshit.dumb3kdb(dering, radius=25, threshold=[48, 48, 48])
ed_grain = adptvgrnMod(ed_deband, strength=0.9, cstrength=0.1, size=1.5, sharp=75, luma_scaling=90)
ed_deband = core.std.MergeDiff(ed_grain, diff)

ending = insert_clip(grain, ed_deband, 30451)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep03/episode3.opus")

final.set_output()

# this is tentative
#vspipe --filter-time Lupin03.vpy - -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -1 --crqpoffs -1 --qcomp 0.7 --rc-lookahead 120 --crf 15.5 --no-sao --aq-mode 3 --aq-strength 0.85 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 33,216,b=0.9/2374,4294,b=0.9/4961,5044,q=7/30449,32127,b=0.95 - -o /mnt/HDD1/lupin/ep03/final.hevc
