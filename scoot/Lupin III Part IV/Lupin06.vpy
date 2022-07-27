import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import mvsfunc as mvf
import G41Fun as G41
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split
from filter import to_rgbs, to_yuvps, texture_mask
from adptvgrnMod import adptvgrnMod
from zzfunc.std import LevelsM
from acsuite import eztrim
from vsdpir import DPIR

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(2446, 4530)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (3255, 3386), (26014, 26145)]
stronger_deband_ranges = [
	(6145, 6252), (14400, 14567), (15243, 15362), 
    (15627, 15674), (15906, 15965), (16446, 16699), (18547, 18753),
    (25403, 25530), (29970, 30056), (32276, 32668)
]

denoise_exclude = OP

starved_grain = [(8524, 10311), (20005, 20162), (20727, 21239), (24397, 25032)]

dpir_ranges = [(4561, 4653), (4762, 4995), (5440, 5841), (6253, 6444)] + starved_grain

no_crepair = OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep06/00004.m2ts")
src = depth(gt, 16)[:32848]

# denoise/deblock
uv_denoise = lvf.rfs(src, core.knlm.KNLMeansCL(src, d=3, a=4, h=1, channels='UV', device_type='gpu'), OP + [(30449, 32127)])

denoise = haf.SMDegrain(uv_denoise, tr=4, thSAD=170, thSADC=100, RefineMotion=True)
denoise = lvf.denoise.bm3d(uv_denoise, ref=denoise, sigma=1.7, final_args={"profile":"fast"})

pre = core.dfttest.DFTTest(src, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.0075, brz_b=0.01)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (0.6, 3.5, 9, 13)])

dpir = to_yuvps(DPIR(to_rgbs(denoise), strength=20, task="deblock", device_type="cuda"))

denoise = lvf.rfs(denoise, dpir, dpir_ranges)
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)
crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = vdf.aa.upscaled_sraa(get_y(crepair), rfactor=1.45)
sraa = join([sraa, plane(crepair, 1), plane(crepair, 2)])

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=50)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[36, 32, 32], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=26, threshold=[60, 60, 60], grain=0)
deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges) 
deband = lvf.rfs(deband, aa, (17864, 17890))

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.15, cstrength=0.075, size=1.2, sharp=65, luma_scaling=20, seed=1)
grain = adptvgrnMod(grain, strength=0.15, cstrength=0.1, size=1, sharp=50, luma_scaling=10, static=True, seed=2)
grain = lvf.rfs(
	grain, 
	core.noise.Add(core.noise.Add(deband, var=3, every=3, type=2, xsize=5, ysize=5), var=3, every=2, type=2, xsize=4, ysize=4), 
	starved_grain
) 

# ED scenefilter

ed = denoise[30449:32129]

dering = to_yuvps(core.w2xnvk.Waifu2x(to_rgbs(ed), noise=3, scale=1, model=2))

nced = core.std.Crop(nced, top=2)[:1678]
nced = core.std.StackVertical([
	nced,
	core.std.Crop(ed, top=1078)
])

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

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep06/00004.flac")

final.set_output()
#vspipe --filter-time -c y4m Lupin06.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 2446,4530,b=0.9 - -o /mnt/HDD1/lupin/ep06/final.hevc
