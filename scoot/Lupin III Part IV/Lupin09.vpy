import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import kagefunc as kgf
import awsmfunc as awf
import lvsfunc as lvf
import G41Fun as G41
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split
from adptvgrnMod import adptvgrnMod
from zzfunc.std import LevelsM
from acsuite import eztrim
from vsdpir import DPIR
from rekt import rektlvls

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(2949, 5034)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (1992, 2063)] 

weakest_deband_ranges = [
		(8787, 10598), (10671, 11755), (11837, 12192),
		(12235, 12432), (12454, 14026), (14075, 15437),
		(15666, 16488), (18440, 18607), (19157, 19533),
		(21477, 21491), (22959, 23030), (24667, 24684),
		(24895, 25020), (28165, 32262)
]

weaker_deband_ranges = [
		(10599, 10670), (15437, 15569), (17958, 18034),
		(19791, 19940), (21300, 21347), (27787, 27858)
]

dpir_ranges = [
		(2500, 2606), (5902, 5961), (10599, 10670),
		(11714, 11755), (12872, 12943), (15200, 15241),
		(17958, 19032), (19431, 19478), (19791, 19940), 
		(21477, 21503), (22581, 22640), (22959, 23060), 
		(23311, 23388), (23626, 23733), (24196, 24810),
		(26710, 27930), (30227, 30352), (32716, None)
]

denoise_exclude = OP

crepair_softer = [(0, 1)]
crepair_harder = [(32716, None)]
no_crepair = OP + dpir_ranges

text_ranges = [(852, 894), (1194, 1251), (1860, 1909), (5035, 5101)]

def to_rgbs(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    clip = depth(clip, 32).std.SetFrameProp('_Matrix', intval=matrix)
    clip = core.resize.Bicubic(clip, format=vs.RGBS)
    return clip

def to_yuvps(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    return core.resize.Bicubic(clip, format=vs.YUV420P16, matrix=matrix)

def texture_mask(clip, radius, points=[x * 256 for x in (1, 2.5, 5, 15)]):
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

gt = core.lsmas.LWLibavSource(f"{filepath}ep09/00004.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt = rektlvls(src, colnum=[1, 1918], colval=[25, 25])
cf = core.cf.ContinuityFixer(rekt, left=[1, 1, 1], right=[1, 1, 1], radius=[2, 2, 2])
bb = awf.bbmod(cf, blur=100, left=4, right=4, y=False)

rekt2 = rektlvls(src, colnum=[0, 1, 1915, 1916, 1917, 1918, 1919], colval=[75, 25, 2, -1, 2, 27, 76])
bb2 = awf.bbmod(rekt2, top=3, bottom=3)
bb = lvf.rfs(bb, bb2, (32716, None))

# denoise/deblock
uv_denoise = lvf.rfs(bb, core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'), OP + [(30449, 32129)])

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=25, task="deblock", device_type="cuda"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=3, thSAD=220, thSADC=150, thSCD2=60, RefineMotion=True)
bm3d = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1, final_args={"profile":"fast"})
denoise = lvf.rfs(denoise, bm3d, [(20965, 21347), (21768, 21992), (23027, 23030), (23909, 23913), (24317, 24876)]) 

pre = core.dfttest.DFTTest(bb, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.01, brz_b=0.01)
texture = texture_mask(pre, radius=3)

text_dering = dpir 
text_mask = kgf.squaremask(denoise, width=544, height=51, offset_x=687, offset_y=931)
denoise = core.std.MaskedMerge(denoise, text_dering, text_mask)
denoise = lvf.rfs(denoise, text_dering, text_ranges)

denoise = haf.ContraSharpening(denoise, bb, planes=[0, 1, 2])
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair_less = vdf.merge_chroma(denoise, dpir)

w2x = to_yuvps(core.w2xc.Waifu2x(to_rgbs(denoise), scale=1, noise=1))
crepair = vdf.merge_chroma(denoise, w2x)

w2xh = to_yuvps(core.w2xc.Waifu2x(to_rgbs(denoise), scale=1, noise=2))
crepair_more = vdf.merge_chroma(denoise, w2xh)

crepair = lvf.rfs(crepair, denoise, no_crepair)
crepair = lvf.rfs(crepair, crepair_less, crepair_softer)
crepair = lvf.rfs(crepair, crepair_more, crepair_harder)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = vdf.aa.upscaled_sraa(get_y(crepair), rfactor=1.4)
sraa = join([sraa, plane(crepair, 1), plane(crepair, 2)])

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=10)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32], grain=0)
cringe = deband

strong_deband = debandshit.dumb3kdb(aa, radius=19, threshold=[48, 48, 48], grain=0)

stronger_deband_luma = core.placebo.Deband(aa, iterations=2, threshold=5, radius=24, grain=0, planes=1)
stronger_deband = core.placebo.Deband(stronger_deband_luma, iterations=2, threshold=4.5, radius=24, grain=0, planes=2|4)
nuke = core.placebo.Deband(stronger_deband_luma, iterations=2, threshold=5, radius=24, grain=0, planes=2|4)
stronger_deband = lvf.rfs(stronger_deband, nuke, (23770, 23913))
deband = lvf.rfs(stronger_deband, deband, weakest_deband_ranges)
deband = lvf.rfs(deband, strong_deband, weaker_deband_ranges) 

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.15, cstrength=0.05, size=1.15, sharp=60, luma_scaling=15, seed=1)
grain = adptvgrnMod(grain, strength=0.15, cstrength=0.05, size=1.1, sharp=55, luma_scaling=9, seed=2, static=True)

# ED scenefilter

ed = uv_denoise[30449:32127]

ed_denoise = haf.SMDegrain(ed, thSAD=170, thSADC=150)
ed_denoise = lvf.denoise.bm3d(ed, ref=ed_denoise, sigma=1.7, final_args={"profile":"fast"})

nced = core.std.Crop(nced, top=2)[:1680]
nced = core.std.StackVertical([
	nced,
	core.std.Crop(ed, top=1078)
])

dering = to_yuvps(core.w2xc.Waifu2x(to_rgbs(ed), scale=1, noise=3))

credit_mask = get_y(core.std.MakeDiff(ed, nced))
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 8)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ed_deband = debandshit.dumb3kdb(ed_denoise, radius=25, threshold=[48, 48, 48])
ed_deband = core.std.MaskedMerge(ed_deband, ed_denoise, detail_mask[30449:32129])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0)
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30449)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep09/00004.flac")

final.set_output()


#vspipe --filter-time -c y4m Lupin07.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 2446,4530,b=0.9 - -o /mnt/HDD1/lupin/ep09/final.hevc
