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

OP = [(1702, 3620)]
opstart = OP[0][0]
starved_grain = [(16590, 16673), (16812, 16932), (20095, 20179)]

aa_include = [
	(opstart+800, opstart+940), (22585, 22675)
] 

stronger_deband_ranges = [
	(1370, 1533), (5798, 5845), (18334, 18411),
	(18502, 18713), (20437, 20580), (20955, 21242),
	(23021, 25138), (25999, 26304), (26536, 26619),
	(26968, 27844), (29150, 29654), (32319, 32399)
]

strongest_deband_ranges = [
	(3786, 3893), (16272, 16535), (17357, 17454),
	(18714, 18773), (20701, 20954), (22501, 22584),
	(28676, 28975), (29654, 29689), (30242, 30313),
	(32415, 32506)
]

dpir_ranges = [
	(6002, 6313), (14586, 14957), (15102, 15197),
	(18502, 18549), (19738, 19773), (20701, 20954), 
	(22501, 22584), (23826, 23957), (24079, 24173),
	(26209, 26304), (26536, 26619), (27034, 27069),
	(29654, 29689), (30242, 30313), (32415, 32437),
	(32616, 32656), (32716, None)
] + starved_grain

denoise_exclude = OP

no_crepair = [
	(529, 727)
] + OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)
nced = nced.knlm.KNLMeansCL(d=3, a=4, h=1, channels='UV', device_type='gpu')

gt = core.lsmas.LWLibavSource(f"{filepath}ep18/00004.m2ts")
src = depth(gt, 16)[:32845]

# edgefix
rekt2 = rektlvls(src, colnum=[1915, 1916, 1917], colval=[1, -1, 2], rownum=[2], rowval=[2])
bb = awf.bbmod(rekt2, top=2, bottom=3, right=2)
bb = lvf.rfs(src, bb, (32716, None))

# denoise/deblock
pre = core.dfttest.DFTTest(bb, sigma=5)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.009, brz_b=0.009)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (1, 1.5, 10, 15)])

uv_denoise = lvf.rfs(
	bb,
	core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP + [(30446, 32124)]
)

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=25, task="deblock"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=6, thSAD=200)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=2, final_args={"profile":"fast"})

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = lvf.sraa(get_y(crepair), rfactor=1.5)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=3)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 38, 38], grain=0)

strong_deband = debandshit.dumb3kdb(aa, radius=24, threshold=[56, 48, 48], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=31, threshold=[74, 70, 70], grain=0)

deband = lvf.rfs(deband, stronger_deband, strongest_deband_ranges)
deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 80, 140)], 
    strengths=[(0.43, 0.15), (0.35, 0.15), (0.32, 0.1)], 
    sizes=[1.35, 1.35, 1.3], sharps=[65, 65, 55], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=False), AddGrain(seed=3, constant=True)]
).graining(deband)
grain = lvf.rfs(grain, core.noise.Add(deband, var=3.7, every=1, type=2, xsize=5, ysize=5), starved_grain)

# ED scenefilter

ed = denoise[30446:32124]

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
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30446:32124])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 

ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30446)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32845), audio_file=f"{filepath}ep18/00004.flac")
from stgfunc import output as o

final.set_output()

#vspipe -c y4m Lupin18.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.69 --psy-rd 2.08 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3--zones 1702,3620,b=0.9 - -o /mnt/HDD1/lupin/ep18/final.hevc
