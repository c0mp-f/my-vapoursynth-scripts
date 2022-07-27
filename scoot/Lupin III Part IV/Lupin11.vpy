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

OP = [(3861, 5945)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (9441, 9596), (16443, 16858)] 

stronger_deband_ranges = [
	(2812, 3461), (6282, 6437), (14586, 14630),
	(14871, 15050), (15711, 15752), (16538, 26304),
	(29395, 30378), (32429, 32535)
]

strongest_deband_ranges = [
	(3819, 3860), (9213, 9308), (10221, 10340),
	(13089, 15338), (15579, 15710)
]

dpir_ranges = [
	(3378, 3461), (3819, 3860), (13089, 14309),
	(14871, 15050), (16113, 16160), (17372, 17410),
	(18621, 18647), (19063, 19248), (19429, 19917),
	(21721, 21864), (24572, 24727), (25494, 25589),
	(25969, 26208), (32504, 32534), (32717, None)
]

denoise_exclude = OP

no_crepair = OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep11/00003.m2ts")
src = depth(gt, 16)[:32849]

# edgefix
rekt2 = rektlvls(src, colnum=[1, 1915, 1916, 1917], colval=[2, 1, -1, 2])
bb = awf.bbmod(rekt2, top=3, bottom=3, right=2)
bb = lvf.rfs(src, bb, (32716, None))

# denoise/deblock
pre = core.dfttest.DFTTest(bb, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.008, brz_b=0.008)
texture = texture_mask(pre, radius=3, points=[x * 256 for x in (1, 2.5, 5, 15)])

uv_denoise = lvf.rfs(
	bb,
	core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'),
	OP
)

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=20, task="deblock"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=6, thSAD=180)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=[1.25, 0, 0], final_args={"profile":"fast"})

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=False)

sraa = lvf.sraa(get_y(crepair), rfactor=1.5)

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=10)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32], grain=0)

strong_deband = debandshit.dumb3kdb(aa, radius=24, threshold=[48, 48, 48], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=31, threshold=[60, 60, 60], grain=0)

deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)
deband = lvf.rfs(deband, stronger_deband, strongest_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = deband
grain = Graigasm(
    thrs=[x << 8 for x in (40, 100, 145)], 
    strengths=[(0.35, 0.15), (0.25, 0.1), (0.2, 0.05)], 
    sizes=[1.3, 1.25, 1.25], sharps=[65, 60, 60], 
    grainers=[AddGrain(seed=1, constant=False), AddGrain(seed=2, constant=False), AddGrain(seed=3, constant=True)]
).graining(deband)

# ED scenefilter

ed = denoise[30450:32130]

nced = core.std.Crop(nced, top=2)[:1680]
nced = core.std.StackVertical([
	nced,
	core.std.Crop(ed, top=1078)
])

dering = to_yuvps(core.w2xnvk.Waifu2x(to_rgbs(ed), noise=3, scale=1, model=2))

credit_mask = get_y(core.std.MakeDiff(ed, nced))
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 8)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ed_y, ed_u, ed_v = split(ed.resize.Bicubic(format=vs.YUV444P16))

ed_dehalo = join([
	get_y(ed),
	dehalo_alpha(plane(ed, 1), rx=2, darkstr=0),
	dehalo_alpha(plane(ed, 2), rx=2, darkstr=0)
])
ed_dehalo = core.std.MaskedMerge(ed, ed_dehalo, credit_mask.std.Maximum().std.Maximum())
ed_dehalo = lvf.rfs(ed, ed_dehalo, (1174, None))

ed_deband = debandshit.dumb3kdb(ed, radius=25, threshold=[48, 48, 48])
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30450:32130])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30450)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32849), audio_file=f"{filepath}ep11/00003.flac")

final.set_output()
#vspipe --filter-time -c y4m Lupin11.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.77 --psy-rd 2 --psy-rdoq 2.15 --ipratio 1.35 --pbratio 1.25 --zones 3861,5945,b=0.9 - -o /mnt/HDD1/lupin/ep11/final.hevc 
