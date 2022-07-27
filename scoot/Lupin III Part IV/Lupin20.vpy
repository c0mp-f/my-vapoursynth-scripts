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
from finedehalo import fine_dehalo, dehalo_alpha
from vardefunc.noise import AddGrain, Graigasm
from adptvgrnMod import adptvgrnMod
from vardefunc import YUVPlanes
from acsuite import eztrim
from rekt import rektlvls
from vsdpir import DPIR
from ccd import ccd

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(2566, 4482)]
opstart = OP[0][0]
starved_grain = [
	(11520, 14462), (25259, 25474), (25646, 25789),
	(25958, 26149), (26222, 26293), (27610, 27824)
]

aa_include = [
	(opstart+800, opstart+940), (5377, 5472), 
	(5569, 5892), (15811, 15828), (21049, 21129)
] 

stronger_deband_ranges = [
	(6346, 6405), (20263, 20430), (23969, 24076),
	(24851, 25186), (25790, 25885), (26294, 26449),
	(26684, 26875), (28713, 28772), (28892, 29330),
	(30234, 30448)
]

strongest_deband_ranges = [
	(17234, 17266), (20605, 20772), (20983, 21048),
	(22707, 22730), (24515, 24604)
]

dpir_ranges = [
	(5779, 5892), (6346, 6405), 
	(32716, None)
] + starved_grain

denoise_exclude = OP

no_crepair = [
	(239, 984), (10395, 10508)
] + OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)
nced = nced.knlm.KNLMeansCL(d=3, a=4, h=1, channels='UV', device_type='gpu')

gt = core.lsmas.LWLibavSource(f"{filepath}ep20/00003.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt2 = rektlvls(src, colnum=[1915, 1916, 1917], colval=[1, -1, 2])
bb = awf.bbmod(rekt2, top=3, bottom=2, right=2)
bb = lvf.rfs(src, bb, (32716, None))

# denoise/deblock
pre = core.dfttest.DFTTest(bb, sigma=5)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.009, brz_b=0.009)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (1, 1.5, 10, 15)])

uv_denoise = lvf.rfs(
	bb,
	core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP + [(30449, 32129)]
)

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=23, task="deblock"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=6, thSAD=200)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=2.1, final_args={"profile":"fast"})

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

singlerater = vdf.aa.Eedi3SR(alpha=0.35, beta=0.5, gamma=125)
sraa = vdf.aa.upscaled_sraa(crepair, rfactor=1.5, singlerater=singlerater)

linemask = core.std.Sobel(pre).std.Binarize(150<<8).std.Maximum().std.Inflate()
sraa_masked = core.std.MaskedMerge(crepair, sraa, linemask)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=1.6)
aa = lvf.rfs(crepair, aa, aa_include)
aa = lvf.rfs(aa, sraa_masked, (5779, 5892))

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 28, 28], grain=0)

strong_deband = debandshit.dumb3kdb(aa, radius=24, threshold=[56, 48, 48], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=31, threshold=[70, 64, 64], grain=0)

deband = lvf.rfs(deband, stronger_deband, strongest_deband_ranges)
deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = Graigasm(
    thrs=[x << 8 for x in (45, 70, 135)], 
    strengths=[(0.43, 0.15), (0.35, 0.1), (0.75, 0.1)], 
    sizes=[1.4, 1.35, 1.5], sharps=[69, 70, 85], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=False), AddGrain(seed=3, constant=True)]
).graining(deband)
grain = lvf.rfs(grain, core.noise.Add(deband, var=4.5, every=1, type=2, xsize=5, ysize=5), starved_grain)

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

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep20/00003.flac")
from stgfunc import output as o

final.set_output()

#vspipe -c y4m Lupin20.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock 0:-1 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.72 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.77 --psy-rd 2.2 --psy-rdoq 2.25 --ipratio 1.37 --pbratio 1.27 --zones 2566,4482,b=0.9 - -o /mnt/HDD1/lupin/ep20/final.hevc
