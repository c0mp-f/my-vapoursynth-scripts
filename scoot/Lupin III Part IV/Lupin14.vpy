import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import kagefunc as kgf
import awsmfunc as awf
import lvsfunc as lvf
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

OP = [(2134, 4052)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (29351, 29433), (30307, 30448)] 

stronger_deband_ranges = [
	(1306, 1356), (5241, 5451), (8658, 8897),
	(9183, 9302), (9503, 9558), (11621, 12477),
	(15705, 15944), (23124, 23302), (23834, 23959),
	(24496, 24503)
]

strongest_deband_ranges = [
	(721, 810), (1069, 1254), (4687, 4938),
	(7694, 7711), (8165, 8236), (8453, 8513),
	(13016, 13432), (16047, 16346), (16528, 16668),
	(25537, 25596), (25933, 26231), (29599, 30235)
]

placebo_deband_ranges = [
	(7904, 8092), (11280, 11415), (11468, 11533),
	(24935, 24962), (25338, 25358), (25801, 25932),
	(29551, 29598), (30236, 30306)
]

dpir_ranges = [
	(8658, 8897), (9042, 9116), (9279, 9302),
	(9997, 10056), (10425, 10559), (10770, 10901),
	(24260, 24328), (24496, 24503), (24573, 24589), 
	(32716, None)
]

denoise_exclude = OP

no_crepair = OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)
nced = nced.knlm.KNLMeansCL(d=3, a=4, h=1, channels='UV', device_type='gpu')

gt = core.lsmas.LWLibavSource(f"{filepath}ep14/00003.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt2 = rektlvls(src, colnum=[1, 1915, 1916, 1917], colval=[2, 1, -1, 2])
bb = awf.bbmod(rekt2, top=3, bottom=3, right=2)
bb = lvf.rfs(src, bb, (32716, None))

# denoise/deblock
pre = core.dfttest.DFTTest(bb, sigma=5)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.009, brz_b=0.009)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (0.8, 1.5, 10, 15)])

uv_denoise = lvf.rfs(
	bb,
	core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP + [(30449, 32129)]
)

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=23, task="deblock"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=6, thSAD=215)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1.7, final_args={"profile":"fast"})

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = lvf.sraa(get_y(crepair), rfactor=1.5)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=3)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32], grain=0)

strong_deband = debandshit.dumb3kdb(aa, radius=24, threshold=[44, 44, 44], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=31, threshold=[64, 64, 64], grain=0)

placebo_deband = debandshit.placebo_deband(aa, radius=24, threshold=[5.5, 4.5], iterations=4, grain=0)

deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)
deband = lvf.rfs(deband, stronger_deband, strongest_deband_ranges)
deband = lvf.rfs(deband, placebo_deband, placebo_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 110, 125)], 
    strengths=[(0.5, 0.15), (0.25, 0.1), (0.15, 0.05)], 
    sizes=[1.35, 1.2, 1.2], sharps=[70, 60, 55], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=True), AddGrain(seed=3, constant=True)]
).graining(deband)

# ED scenefilter

ed = denoise[30449:32128]

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
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30449:32128])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 

ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30449)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep14/00003.flac")
from stgfunc import output as o

final.set_output()

#vspipe -c y4m Lupin14.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.76 --psy-rd 2 --psy-rdoq 2.15 --ipratio 1.35 --pbratio 1.25 --zones 2134,4052,b=0.9 - -o /mnt/HDD1/lupin/ep14/final.hevc
