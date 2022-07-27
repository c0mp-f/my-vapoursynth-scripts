import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import kagefunc as kgf
import awsmfunc as awf
import lvsfunc as lvf
import G41Fun as G41
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split, scale_value
from filter import to_rgbs, to_yuvps, texture_mask
from finedehalo import fine_dehalo, dehalo_alpha
from vardefunc.noise import Graigasm, AddGrain
from adptvgrnMod import adptvgrnMod, sizedgrn
from vardefunc import YUVPlanes
from acsuite import eztrim
from rekt import rektlvls
from vsdpir import DPIR
from ccd import ccd

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(2158, 4242)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (15393, 15429), (16482, 16499)]

weaker_deband_ranges = [
    (4502, 6578), (6681, 8549), (8860, 9129), (13000, 16186),
    (16220, 16433), (16601, 16749), (16993, 17061), (17638, 18341),
    (18378, 19786), (19918, 20364), (20449, 21459), (21517, 21606),
    (21817, 21877), (23145, 23233), (23611, 24184), (24362, 24421),
    (24647, 25006), (25190, 25480), (25856, 25990), (27796, 27952), (28106, 30137)
]

denoise_exclude = OP

dpir_ranges = [
    (1085, 1360), (1820, 1933), (9130, 9267), 
    (13153, 13160), (19786, 19917), (20449, 20517), (21310, 21330), 
    (25007, 25189), (25991, 26792), (27953, 28105), (32716, None)
]

text_ranges = [(1766, 1819), (4245, 4285)]

no_crepair = OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep10/00002.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt = rektlvls(src, colnum=[0, 1], colval=[3, 1])
bb2 = awf.bbmod(rekt, top=3, bottom=3, right=4)
bb = lvf.rfs(src, bb2, (32716, None))

# denoise/deblock
uv_denoise = lvf.rfs(
    bb,
	core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP + [(30449, 32127)]
)

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=30, task="deblock", device_type="cuda"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

dft = core.dfttest.DFTTest(bb, sigma=2)
deblock = lvf.rfs(deblock, dft, (29959, 30112))

denoise = haf.SMDegrain(deblock, tr=2, thSAD=220,  plane=0)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1, final_args={"profile":"fast"})

pre = core.dfttest.DFTTest(bb, sigma=8)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.008, brz_b=0.01)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (1, 4, 5, 9)])

text_dering = dpir 
text_mask = kgf.squaremask(denoise, width=983, height=51, offset_x=468, offset_y=931)
denoise = core.std.MaskedMerge(denoise, text_dering, text_mask)
denoise = lvf.rfs(denoise, text_dering, text_ranges)

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = lvf.sraa(get_y(crepair), rfactor=1.7)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=20)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32])

stronger_deband_luma = core.placebo.Deband(aa, iterations=3, threshold=4.8, radius=40, grain=0, planes=1)
stronger_deband = core.placebo.Deband(stronger_deband_luma, iterations=2, threshold=4.8, radius=40, grain=0, planes=2|4)

# there is no detail for me to be worrying about detail loss
nuke = core.placebo.Deband(aa, iterations=4, threshold=9, radius=64, grain=0, planes=1|2|4)

stronger_deband = lvf.rfs(stronger_deband, nuke, (27683, 27754))
deband = lvf.rfs(stronger_deband, deband, weaker_deband_ranges) 

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 80, 130, 170)],
    strengths=[(0.4, 0.05), (0.35, 0.05), (0.25, 0), (0.15, 0)],
    sizes=[1.35, 1.3, 1.15, 1.15], sharps=[65, 60, 55, 50],
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=False), AddGrain(seed=3, constant=False), AddGrain(seed=4, constant=True)]
).graining(deband)
grain = lvf.rfs(
    grain,
    core.noise.Add(core.noise.Add(deband, var=3, every=5, type=2, xsize=5, ysize=5), var=2, every=2, type=2, xsize=5, ysize=5),
    (25856, 25894)
)


# ED scenefilter
ed = denoise[30449:32127]

nced = core.std.Crop(nced, top=2)[:1680]
nced = core.std.StackVertical([
	nced,
	core.std.Crop(ed, top=1078)
])

dering = to_yuvps(core.w2xnvk.Waifu2x(to_rgbs(ed), noise=3, scale=1, model=2))

credit_mask = core.std.MakeDiff(ed, nced)
credit_mask = [credit_mask] * 3
credit_mask = join(credit_mask).resize.Bicubic(format=vs.YUV420P16)
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 8)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ed_dehalo = core.std.Expr([ed, nced, rings], ["", "z x y max x ?", "z x y min x ?"])

ed_deband = debandshit.dumb3kdb(ed, radius=25, threshold=[48, 48, 48])
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30449:32129])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30449)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep10/00002.flac")

from stgfunc import output as o

final.set_output()

#vspipe -c y4m Lupin10.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.75 --rc-lookahead 120 --no-sao --aq-mode 3 --crf 15 --aq-strength 0.7 --psy-rd 2.1 --psy-rdoq 2.1 --ipratio 1.35 --pbratio 1.25 --zones 2158,4242,b=0.9/8907,12809,q=7/23233,23484,q=7/23611,24280,b=1.2/24281,24361,b=1.2/24422,25006,q=8/25007,25189,b=1.1/25895,25990,q=8/27869,27952,q=8/28208,29959,b=1.2  - -o /mnt/HDD1/lupin/ep10/final.hevc
