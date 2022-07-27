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
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsdpir import DPIR
from rekt import rektlvls

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(1104, 3188)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (27248, 27397)]

stronger_deband_ranges = [
    (4863, 5444), (5637, 6104), (7275, 7317),
	(7713, 8276), (8439, 8972), (9141, 9203),
	(9264, 9308), (9360, 9633), (10102, 12529),
	(12620, 17491), (17570, 17809), (17960, 22499),
	(22605, 27722), (28088, 29145), (29977, 32164)
]

denoise_exclude = OP

dpir_ranges = [
	(7713, 8276), (8439, 8558), (9141, 9203),
	(9264, 9308), (9360, 9377), (21725, 22300),
	(22500, 22604), (28088, 28449), (29062, 29349), (32716, None)
]

no_crepair = OP + dpir_ranges

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep08/00003.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt = rektlvls(src, colnum=[1, 1918], colval=[24, 24])
cf = core.cf.ContinuityFixer(rekt, left=[1, 0, 0], right=[1, 0, 0], radius=[2, 0, 0])
bb = awf.bbmod(cf, left=2, right=4, y=False) # 1 left, 2 right

rekt2 = rektlvls(src, colnum=[0, 1, 1916, 1918, 1919], colval=[75, 25, -2, 25, 75])
bb2 = awf.bbmod(rekt2, top=3, bottom=3)
bb = lvf.rfs(bb, bb2, (32716, None))

# denoise/deblock
uv_denoise = lvf.rfs(bb, core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'), OP + [(30449, 32129)])

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=20, task="deblock", device_type="cuda"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

denoise = haf.SMDegrain(deblock, tr=3, thSAD=220, thSADC=150, thSCD2=60, RefineMotion=True)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1, final_args={"profile":"fast"})

pre = core.dfttest.DFTTest(bb, sigma=6)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.009, brz_b=0.01)
texture = texture_mask(pre, radius=4, points=[x * 256 for x in (0.9, 3.5, 5, 15)])
texture = insert_clip(texture, core.std.BlankClip(texture, length=83), 8811)

denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = vdf.merge_chroma(denoise, dpir)
crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = vdf.aa.upscaled_sraa(get_y(crepair), rfactor=1.7)
sraa = join([sraa, plane(crepair, 1), plane(crepair, 2)])

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=20)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 24, 24], grain=0)

stronger_deband = core.placebo.Deband(aa, iterations=3, threshold=4.7, radius=24, grain=0, planes=1)
stronger_deband = core.placebo.Deband(stronger_deband, iterations=2, threshold=5, radius=30, grain=1, planes=2|4)
deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges) 

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.25, cstrength=0.05, size=1.15, sharp=60, luma_scaling=12, seed=1)
grain = lvf.rfs(
	grain,
	core.noise.Add(core.noise.Add(deband, var=3, every=3, type=2, xsize=5, ysize=5), var=3, every=2, type=2, xsize=4, ysize=4),
	(28088, 28449)
)

# ED scenefilter

ed = denoise[30449:32127]

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

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep08/00003.flac")

final.set_output()

#vspipe -c y4m my-vapoursynth-scripts/scoot/Lupin\ III\ Part\ IV/Lupin08.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 1104,3188,b=0.95 - -o /mnt/HDD1/lupin/ep08/final.hevc
