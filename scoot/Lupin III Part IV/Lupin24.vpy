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
from filter import to_rgbs, to_yuvps, texture_mask
from vardefunc.noise import AddGrain, Graigasm
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsdpir import DPIR

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(1894, 3812)]
opstart = OP[0][0]

aa_include = [
	(opstart+800, opstart+940), (7752, 7895), (9909, 9980), (21997, 22090)
] 

strongest_deband_ranges = [
	(5359, 5448), (20698, 20712), (21311, 21352)
]

denoise_exclude = OP

no_crepair = OP

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)
nced = nced.knlm.KNLMeansCL(d=3, a=4, h=1, channels='UV', device_type='gpu')

gt = core.lsmas.LWLibavSource(f"{filepath}ep24/00004.m2ts")
src = depth(gt, 16)[:32128]

# denoise/deblock
pre = core.dfttest.DFTTest(src, sigma=5)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.009, brz_b=0.009)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (1, 1.5, 10, 15)])

uv_denoise = lvf.rfs(
	src,
	core.knlm.KNLMeansCL(src, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP
)

denoise = haf.SMDegrain(uv_denoise, tr=6, thSAD=180)
denoise = lvf.denoise.bm3d(uv_denoise, ref=denoise, sigma=1.7, final_args={"profile":"fast"})

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

dpir = to_yuvps(DPIR(to_rgbs(denoise), strength=18, task="deblock"))
crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = lvf.sraa(crepair, rfactor=1.5)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=1)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 16, 16], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=31, threshold=[70, 60, 60], grain=0)

deband = lvf.rfs(deband, stronger_deband, strongest_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = Graigasm(
    thrs=[x << 8 for x in (45, 90, 140)], 
    strengths=[(0.33, 0.15), (0.25, 0.1), (0.35, 0.05)], 
    sizes=[1.4, 1.35, 1.4], sharps=[70, 60, 75], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=False), AddGrain(seed=3, constant=True)]
).graining(deband)

# ED scenefilter

ed = denoise[30449:32127]

nced = core.std.Crop(nced, top=2)[:1679]
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

#eztrim(gt, trims=(None, 32128), audio_file=f"{filepath}ep24/00004.flac")
from stgfunc import output as o

final.set_output()

#vspipe -c y4m Lupin24.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -1:-1 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.71 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.1 --psy-rdoq 2.2 --ipratio 1.4 --pbratio 1.3 --zones 1894,3812,b=0.9 - -o /mnt/HDD1/lupin/ep24/final.hevc
