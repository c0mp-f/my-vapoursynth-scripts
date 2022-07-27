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
	(19791, 19940), (21300, 21398), (27787, 27858)
]

dpir_ranges = [
	(2500, 2606), (5902, 5961), (10599, 10670),
	(11714, 11755), (12872, 12943), (15200, 15241),
	(17958, 19032), (19431, 19478), (19791, 19940), 
	(21477, 21503), (22581, 22640), (22959, 23060), 
	(23311, 23388), (23626, 23733), (23770, 23913), 
    (24196, 24810), (26710, 27930), (30227, 30352), 
    (32716, None)
]

denoise_exclude = OP

crepair_softer = [(0, 1)]
no_crepair = OP + dpir_ranges

text_ranges = [(852, 894), (1194, 1251), (1860, 1909), (5035, 5101)]

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep09/00004.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt = rektlvls(src, colnum=[1, 1918], colval=[25, 25])
cf = core.cf.ContinuityFixer(rekt, left=[1, 1, 1], right=[1, 1, 1], radius=[2, 2, 2])
bb = awf.bbmod(cf, blur=100, left=6, right=4, y=False)

rekt2 = rektlvls(src, colnum=[0, 1, 1915, 1916, 1917, 1918, 1919], colval=[75, 25, 2, -1, 2, 27, 76])
bb2 = awf.bbmod(rekt2, top=3, bottom=3)
bb = lvf.rfs(bb, bb2, (32716, None))
bb = bb.resize.Bicubic(format=vs.YUV444P16)

# denoise/deblock
pre = core.dfttest.DFTTest(bb, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.01, brz_b=0.01)
detail_masks = [lvf.mask.detail_mask(plane, rad=4, brz_a=0.006, brz_b=0.006) for plane in split(pre.resize.Bicubic(format=vs.RGB48, matrix_in=1)[23770:23993])]
detail_mask = insert_clip(detail_mask, core.std.Expr(detail_masks, "x y z max max"), 23770)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (1, 2.5, 5, 15)])

uv_denoise = lvf.rfs(
	ccd(bb, threshold=13),
	core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP
)

dpir = DPIR(to_rgbs(bb), strength=20, task="deblock").resize.Bicubic(format=vs.YUV444P16, matrix=1)
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=2, thSAD=220, plane=0)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=[0.75, 0, 0], final_args={"profile":"fast"})

text_dering = dpir 
text_mask = kgf.squaremask(denoise, width=544, height=51, offset_x=687, offset_y=931)
denoise = core.std.MaskedMerge(denoise, text_dering, text_mask)
denoise = lvf.rfs(denoise, text_dering, text_ranges)

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair_magic = vdf.merge_chroma(denoise, dpir)

with YUVPlanes(denoise) as c:
	def _dehalo(clip: vs.VideoNode) -> vs.VideoNode:
		denoise = lvf.denoise.bm3d(clip, sigma=9.5, radius=0, basic_args={"profile":"np"})

		ref = core.bilateral.Bilateral(denoise, sigmaS=1.5, sigmaR=7/255)
		bidh = core.bilateral.Bilateral(denoise, ref=ref, sigmaS=0.5, sigmaR=5/255)

		minp = core.std.Expr([clip, bidh], "x y min")

		return fine_dehalo(bidh, minp, 2.2, 2.2, 0, 1.2, 60, 80)

	c.U = _dehalo(c.U)
	c.V = _dehalo(c.V)
crepair = c.clip

crepair = crepair.warp.AWarpSharp2(
	thresh=128, blur=2, type=1, depth=12, planes=[1, 2]
)
crepair = lvf.rfs(crepair, denoise, no_crepair)
crepair = lvf.rfs(crepair, crepair_magic, crepair_softer)
crepair = crepair.resize.Bicubic(format=vs.YUV420P16)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = lvf.sraa(get_y(crepair), rfactor=1.4)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=10)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32], grain=0)

strong_deband = debandshit.dumb3kdb(aa, radius=19, threshold=[48, 48, 48], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=31, threshold=[60, 60, 60], grain=0)

nuke = core.placebo.Deband(aa, iterations=2, threshold=3.3, radius=30, grain=0, planes=1)
nuke = core.placebo.Deband(nuke, iterations=2, threshold=1.6, radius=30, grain=0, planes=2|4)

stronger_deband = lvf.rfs(stronger_deband, nuke, (23770, 23913))

deband = lvf.rfs(stronger_deband, deband, weakest_deband_ranges)
deband = lvf.rfs(deband, strong_deband, weaker_deband_ranges) 

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 100, 155)], 
    strengths=[(0.35, 0.15), (0.25, 0.1), (0.2, 0.05)], 
    sizes=[1.2, 1.15, 1.15], sharps=[65, 60, 60], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=False), AddGrain(seed=3, constant=True)]
).graining(deband)

# ED scenefilter

ed = crepair[30449:32127]

nced = core.std.Crop(nced, top=2)[:1680]
nced = core.std.StackVertical([
	nced,
	core.std.Crop(ed, top=1078)
])

dering = to_yuvps(core.w2xc.Waifu2x(to_rgbs(ed), noise=3, scale=1))

credit_mask = get_y(core.std.MakeDiff(ed, nced))
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 8)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ed_dehalo = join([
	get_y(ed),
	dehalo_alpha(plane(ed, 1), rx=2, darkstr=0),
	dehalo_alpha(plane(ed, 2), rx=2, darkstr=0.3)
])
ed_dehalo = core.std.MaskedMerge(ed, ed_dehalo, credit_mask.std.Maximum().std.Maximum())

ed_deband = debandshit.dumb3kdb(ed_dehalo, radius=25, threshold=[48, 48, 48])
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30449:32129])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30449)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep09/00004.flac")

#src.set_output()

final.set_output()
src.set_output(1)

sp(src).set_output(2)
sp(final).set_output(3)

#vspipe --filter-time -c y4m Lupin09.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 2446,4530,b=0.9 - -o /mnt/HDD1/lupin/ep09/final.hevc
