import vapoursynth as vs
import vardefunc as vdf
import awsmfunc as awf
import havsfunc as haf
import lvsfunc as lvf
import G41Fun as G41
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split
from filter import to_rgbs, to_yuvps, texture_mask
from vardefunc import Graigasm, AddGrain
from adptvgrnMod import adptvgrnMod
from finedehalo import fine_dehalo
from acsuite import eztrim
from vsdpir import DPIR

core = vs.core
filepath = "/mnt/HDD1/lupin/"

# OP/ED
OP = [(1416, 3500)]
opstart = OP[0][0]

aa_include = [
    (opstart+800, opstart+940), (5262, 5893), (6579, 6612), 
    (7225, 8990), (9111, 9356), (14893, 15036), 
    (18614, 18655), (22523, 22690), (30377, 30448)
]

stronger_deband_ranges = [
    (5479, 6620), (7039, 7098), (7615, 7710), 
    (8566, 8792), (10158, 10217), (12037, 13124), 
    (13389, 13448), (15109, 15204), (18482, 19513), 
    (19680, 19706), (28243, 29148), (32603, 32656)
]

dpir_ranges = [(32716, None)]

denoise_exclude = OP

no_crepair = OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)
nced = nced.knlm.KNLMeansCL(d=3, a=4, h=1, channels='UV', device_type='gpu')

gt = core.lsmas.LWLibavSource(f"{filepath}ep01/00002.m2ts")
src = core.std.Crop(gt, top=2)[:32848]
src = depth(src, 16)

# edgefixering
bb = awf.bbmod(src, top=3, bottom=1, left=2, right=4)
bb = lvf.rfs(src, bb, (32716, None))

# denoise/deblock
uv_denoise = core.knlm.KNLMeansCL(bb, d=3, a=4, h=0.5, channels='UV', device_type='gpu')
uv_denoise = lvf.rfs(bb, uv_denoise, OP + [(30449, 32127)])

dpir = to_yuvps(DPIR(to_rgbs(uv_denoise), strength=20, task="deblock"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, thSAD=200)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1.7, final_args={"profile":"fast"})
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = dpir
crepair = vdf.merge_chroma(denoise, crepair)
crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)
sraa = lvf.sraa(crepair, rfactor=1.5)

clamp = lvf.aa.clamp_aa(crepair, weak=transpose, strong=sraa, strength=10)
aa = lvf.rfs(crepair, clamp, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32], grain=0)

detail_mask = lvf.mask.detail_mask(denoise, rad=4, brz_a=0.01, brz_b=0.02)
texture = texture_mask(denoise, radius=3)

strong_deband = debandshit.dumb3kdb(aa, radius=25, threshold=[60, 44, 44])
deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

cope = texture_mask(denoise, radius=10)
sceneflt_deband = core.std.MaskedMerge(deband, aa, cope)
deband = lvf.rfs(deband, sceneflt_deband, [(8712, 8792), (18614, 18655)])

grain = Graigasm(
    thrs=[x << 8 for x in (40, 80, 125, 150)], 
    strengths=[(0.35, 0.15), (0.3, 0.1), (0.25, 0.05), (0.25, 0)], 
    sizes=[1.3, 1.2, 1.2, 1.2], sharps=[70, 70, 60, 60], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=True), AddGrain(seed=3, constant=True), AddGrain(seed=4, constant=True)]
).graining(deband)

# ED scenefilter
# Tons of ringing, on text most noticably :|

ed = denoise[30449:32127]

# make the same length and resolution
nced = core.std.Crop(nced, top=2)[:1678]

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

eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep01/00002.flac")

final.set_output()

#vspipe --filter-time Lupin01.vpy - -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --no-open-gop --no-cutree --rd 3 --merange 57 --high-tier --range limited --aud --repeat-headers --bframes 16 --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.72 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2.15 --ipratio 1.35 --pbratio 1.25 --zones 31,239,b=0.85/1416,3500,b=0.9/3633,3707,q=7 - -o /mnt/HDD1/lupin/ep01/final.hevc
