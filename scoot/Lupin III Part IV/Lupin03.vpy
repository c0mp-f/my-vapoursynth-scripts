import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
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

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(2374, 4294)]
opstart = OP[0][0]

aa_include = [(opstart+800, opstart+940), (11355, 11480)]
stronger_deband_ranges = [
    (4572, 4621), (9709, 9846), (10860, 11912), 
	(12185, 12593), (16515, 16555), (17889, 17928), 
	(18351, 19463), (24585, 25617), (25977, 26624), 
	(29769, 29942), (29943, 30166)
]

placebo_deband_ranges = [(650, 2373), (4261, 6218), (7656, 9018), (9967, 10030), (16785, 16818), (20051, 20146)]
denoise_exclude = [(2374, 4460), (12097, 12183)]
crepair_range = [
    (4741, 8544), (8599, 9127), (9444, 9708), 
	(9847, 10188), (10668, 10715), (10860, 12097), 
	(14566, 16422), (16554, 25616), (25977, 26624), 
	(26885, 30400), (32671, None)
]

unfuck_harder = [(13906, 14380), (15284, 15370), (15443, 15682), (17536, 17912), (19485, 21685), (22034, 26624), (26885, 28526)]
deblock_necessary = [(9847, 9966), (4815, 5044), (11091, 11354), (11481, 11588), (29953, 30322), (30239, 30442)]

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)
nced = nced.knlm.KNLMeansCL(d=3, a=4, h=1, channels='UV', device_type='gpu')

gt = core.lsmas.LWLibavSource(f"{filepath}ep03/00004.m2ts")
src = core.std.Crop(gt, top=2)[:32848]
src = depth(src, 16)

# edgefixing
bb = awf.bbmod(src, top=2, left=2, right=2, bottom=1)
bb = lvf.rfs(src, bb, (32671, None))

# denoise/deblock
uv_denoise = lvf.rfs(bb, core.knlm.KNLMeansCL(bb, d=3, a=4, h=0.75, channels='UV', device_type='gpu'), OP + [(30451, 32129)])

deblock = core.dfttest.DFTTest(bb, sigma=6)

denoise = haf.SMDegrain(deblock, tr=6, thSAD=200)
denoise = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1.7, final_args={"profile":"fast"})
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)
denoise = lvf.rfs(denoise, deblock, deblock_necessary)

pre = core.dfttest.DFTTest(bb, sigma=10)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.02, brz_b=0.005)
texture = texture_mask(pre, radius=6)

crepair = to_yuvps(DPIR(to_rgbs(denoise), strength=20, task="deblock"))
crepair = vdf.merge_chroma(denoise, crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

singlerater = vdf.aa.Eedi3SR({"alpha":0.5, "beta": 0.4, "gamma":80})
sraa = vdf.aa.upscaled_sraa(get_y(crepair), rfactor=1.1, width=1920, height=1078, singlerater=singlerater)
sraa = join([sraa, plane(crepair, 1), plane(crepair, 2)])

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=1)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[40, 32, 32], grain=0)

deband = lvf.rfs(deband, denoise, denoise_exclude)
stronger_deband = debandshit.dumb3kdb(aa, radius=26, threshold=[60, 60, 60])
deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges) 

placebo_deband = core.placebo.Deband(aa, radius=18, iterations=1, threshold=6, grain=0, planes=1, dither=0)
placebo_deband = core.placebo.Deband(placebo_deband, radius=18, iterations=1, threshold=7, grain=0, planes=2|4, dither=0)
deband = lvf.rfs(deband, placebo_deband, placebo_deband_ranges)

deband = lvf.rfs(deband, debandshit.dumb3kdb(aa, radius=19, threshold=[32, 12, 12]), [(4961, 5044), (9847, 9966)])

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.24, cstrength=0.1, size=1.25, sharp=69, luma_scaling=24, seed=1)
grain = adptvgrnMod(grain, strength=0.1, cstrength=0.05, size=1, sharp=55, luma_scaling=10, static=True, seed=2)

# ED scenefilter

ed = denoise[30451:32129]

dering = to_yuvps(core.w2xnvk.Waifu2x(to_rgbs(ed), noise=3, scale=1, model=2))

nced = core.std.Crop(nced, top=2)[:1678]

credit_mask = core.std.MakeDiff(ed, nced)
credit_mask = [credit_mask] * 3
credit_mask = join(credit_mask).resize.Bicubic(format=vs.YUV420P16)
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 8)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ed_dehalo = core.std.Expr([ed, nced, rings], ["", "z x y max x ?", "z x y min x ?"])

ed_deband = debandshit.dumb3kdb(ed, radius=25, threshold=[48, 48, 48])
ed_deband = core.std.MaskedMerge(ed_deband, ed_dehalo, detail_mask[30451:32129])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0) 
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30451)

final = depth(ending, 10)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep03/00004.flac")

final.set_output()
src.set_output(1)

#vspipe Lupin03.vpy --filter-time - -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -1 --crqpoffs -1 --qcomp 0.7 --rc-lookahead 120 --crf 15.5 --no-sao --aq-mode 3 --aq-strength 0.77 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 33,216,b=0.9/2374,4294,b=0.9/4961,5044,q=7 - -o /mnt/HDD1/lupin/ep03/final.hevc