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
from rekt import rektlvls
from vsdpir import DPIR

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(1224, 3308)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (18513, 18621), (27291, 27377), (30173, 30268)]
stronger_deband_ranges = [
    (7770, 7956), (10781, 11440), (14352, 14580), (16379, 16660), 
    (16937, 16960), (17495, 17554), (20016, 20099), (22740, 22811),
    (25328, 26919), (27019, 27114), (29062, 30172)
]

denoise_exclude = OP
no_crepair = OP

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep04/00002.m2ts")
src = depth(gt, 16)[:32848]

dering = to_yuvps(DPIR(to_rgbs(src), strength=25, task="deblock", device_type="cuda"))
bb2 = awf.bbmod(dering, top=3, bottom=3, left=3, right=3)
src = lvf.rfs(src, bb2, (32716, None))

# denoise/deblock
uv_denoise = lvf.rfs(src, core.knlm.KNLMeansCL(src, d=3, a=4, h=0.75, channels='UV', device_type='gpu'), OP + [(30449, 32129)])

denoise = haf.SMDegrain(uv_denoise, tr=3, thSAD=200)
denoise = lvf.denoise.bm3d(uv_denoise, ref=denoise, sigma=1.7, final_args={"profile":"fast"})
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

pre = core.dfttest.DFTTest(src, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.0075, brz_b=0.025)
texture = texture_mask(pre, radius=4)

dpir = dering
crepair = vdf.merge_chroma(denoise, dpir)

crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = vdf.aa.upscaled_sraa(get_y(crepair), rfactor=1.3)
sraa = join([sraa, plane(crepair, 1), plane(crepair, 2)])

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=1)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[32, 32, 32], grain=0)

stronger_deband = debandshit.dumb3kdb(aa, radius=26, threshold=[60, 60, 60])
deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges) 
deband = lvf.rfs(deband, debandshit.dumb3kdb(aa, radius=26, threshold=[60, 60, 60]), (16961, 17098))

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.2, cstrength=0.1, size=1.2, sharp=69, luma_scaling=12, seed=1)
grain = adptvgrnMod(grain, strength=0.1, cstrength=0.1, size=0.9, sharp=55, luma_scaling=8, static=True, seed=2)

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
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30449:32127])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30449)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep04/00002.flac")
final.set_output()
src.set_output(1)

#vspipe Lupin04.vpy - -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 1224,3308,b=0.9/7771,7860,q=7 - -o /mnt/HDD1/lupin/ep04/final.hevc
