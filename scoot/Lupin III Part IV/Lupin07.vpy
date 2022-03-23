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
from adptvgrnMod import adptvgrnMod
from zzfunc.std import LevelsM
from acsuite import eztrim
from vsdpir import DPIR
from rekt import rektlvls

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

OP = [(1870, 3956)]

aa_include = [(2680, 2810), (18367, 18435), (27492, 27617)]

stronger_deband_ranges = [
        (1005, 1076), (11076, 11183), (11328, 11372), (11544, 12986),
        (13248, 13572), (13849, 13992), (14265, 14336), (18367, 18435), 
        (30262, 30448)
]

placebo_deband_ranges = [
        (4103, 4718), (12597, 13199), (13507, 13530), (13992, 14264),
        (16252, 16443), (16999, 17610), (19024, 19063), (19133, 19321),
        (20319, 20446), (21174, 21461), (21934, 26224), (26781, 27491),
        (29353, 29456), (29617, 29679), (29781, 29948), (32321, 32418),
        (32521, 32694)
]

denoise_exclude = OP + [(5553, 5600), (9885, 10013), (10062, 10245)]

crepair_softer = [(18319, 18366), (20447, 20493), (20650, 20689), (30189, 30251)]
crepair_harder = [(27797, 28579)]

dpir_ranges = [
        (426, 515), (612, 659), (1005, 1076), (13116, 13199),
        (13483, 13584), (16060, 16179), (16384, 16569),
        (18367, 18435), (30377, 30448)
]

no_crepair = OP + dpir_ranges

def to_rgbs(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    clip = depth(clip, 32).std.SetFrameProp('_Matrix', intval=matrix)
    clip = core.resize.Bicubic(clip, format=vs.RGBS)
    return clip

def to_yuvps(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    return core.resize.Bicubic(clip, format=vs.YUV420P16, matrix=matrix)

def texture_mask(clip, radius, points=[x * 256 for x in (0.9, 1.75, 8, 10)]):
     ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
     rmask = vdf.mask.MinMax(radius).get_mask(get_y(clip), lthr=0, multi=1.00)
     emask = ed_gray.std.Prewitt()
     em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
     em_hi = iterate(em_hi, core.std.Minimum, 5)
     em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
     em_me = iterate(em_me, core.std.Minimum, 4)
     em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
     em_lo = iterate(em_lo, core.std.Minimum, 2)
     rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
     weighted = LevelsM(rm_txt, points=points, levels=[0, 1, 1, 0], xpass=[0, 0], return_expr=0)

     mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535*0.2} - {1 / (1 - 0.2)} *')
     return mask

# sources
NCED = core.lsmas.LWLibavSource(f"{filepath}NCED.m2ts")
nced = depth(NCED, 16)

gt = core.lsmas.LWLibavSource(f"{filepath}ep07/00002.m2ts")
src = depth(gt, 16)[:32848]

# edgefix
rekt = rektlvls(src, colnum=[1, 1917, 1918], colval=[24, -1, 24])
cf = core.cf.ContinuityFixer(rekt, left=[1, 2, 2], right=[1, 2, 2], radius=[2, 2, 2])

rekt2 = rektlvls(src, colnum=[0, 1, 1916, 1918, 1919], colval=[75, 25, -2, 25, 75])
bb2 = awf.bbmod(rekt2, top=3, bottom=3)
bb = lvf.rfs(cf, bb2, (32716, None))

# denoise/deblock
uv_denoise = lvf.rfs(bb, core.knlm.KNLMeansCL(bb, d=3, a=4, h=1, channels='UV', device_type='gpu'), OP + [(30449, 32129)])

dpir = to_yuvps(DPIR(to_rgbs(bb), strength=17, task="deblock", device_type="cuda"))
deblock = lvf.rfs(uv_denoise, dpir, dpir_ranges)

dft = core.dfttest.DFTTest(uv_denoise, sigma=4)
deblock = lvf.rfs(deblock, dft, (12978, 13061))

denoise = haf.SMDegrain(deblock, tr=6, thSAD=230, thSADC=150, RefineMotion=True)
bm3d = lvf.denoise.bm3d(deblock, ref=denoise, sigma=1, final_args={"profile":"np"})
denoise = lvf.rfs(denoise, bm3d, (19400, 26224))
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

pre = core.dfttest.DFTTest(bb, sigma=6)
detail_mask = lvf.mask.detail_mask(pre, rad=3, brz_a=0.009, brz_b=0.01)
texture = texture_mask(pre, radius=3)

crepair_less = vdf.merge_chroma(denoise, dpir)

w2x = to_yuvps(core.w2xc.Waifu2x(to_rgbs(denoise), scale=1, noise=1))
crepair = vdf.merge_chroma(denoise, w2x)

w2xh = to_yuvps(core.w2xc.Waifu2x(to_rgbs(denoise), scale=1, noise=2))
crepair_more = vdf.merge_chroma(denoise, w2xh)

crepair = lvf.rfs(crepair, crepair_less, crepair_softer)
crepair = lvf.rfs(crepair, crepair_more, crepair_harder)
crepair = lvf.rfs(crepair, denoise, no_crepair)

# AA
transpose = lvf.aa.transpose_aa(crepair, eedi3=True)

sraa = vdf.aa.upscaled_sraa(get_y(crepair), rfactor=1.5)
sraa = join([sraa, plane(crepair, 1), plane(crepair, 2)])

aa = lvf.aa.clamp_aa(crepair, transpose, sraa, strength=40)
aa = lvf.rfs(crepair, aa, aa_include)

# deband
deband = debandshit.dumb3kdb(aa, radius=19, threshold=[45, 45, 45], grain=0)
deband1 = deband

stronger_deband = debandshit.dumb3kdb(aa, radius=26, threshold=[60, 60, 60], grain=0)
deband2 = stronger_deband
deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges) 

placebo = core.placebo.Deband(aa, iterations=2, threshold=5, radius=24, grain=1, planes=1)
placebo = core.placebo.Deband(placebo, iterations=1, threshold=7, radius=24, grain=0, planes=2|4)
deband = lvf.rfs(deband, placebo, placebo_deband_ranges)

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.std.MaskedMerge(deband, aa, texture)

grain = adptvgrnMod(deband, strength=0.2, cstrength=0.05, size=1.15, sharp=60, luma_scaling=18, seed=1)
grain = adptvgrnMod(grain, strength=0.15, cstrength=0.05, size=1.1, sharp=50, luma_scaling=8, static=True, seed=2)

# ED scenefilter

ed = uv_denoise[30449:32127]

ed_denoise = haf.SMDegrain(ed, thSAD=170, thSADC=150)
ed_denoise = lvf.denoise.bm3d(ed, ref=ed_denoise, sigma=1.7, final_args={"profile":"fast"})

nced = core.std.Crop(nced, top=2)[:1680]
nced = core.std.StackVertical([
	nced,
	core.std.Crop(ed, top=1078)
])

dering = to_yuvps(core.w2xc.Waifu2x(to_rgbs(ed), scale=1, noise=3))

credit_mask = get_y(core.std.MakeDiff(ed, nced))
credit_mask = core.std.Binarize(credit_mask, 38000)
expanded_mask = iterate(credit_mask, core.std.Maximum, 8)
rings = core.std.Expr([credit_mask, expanded_mask], "x y - abs")

ed_deband = debandshit.dumb3kdb(ed_denoise, radius=25, threshold=[48, 48, 48])
ed_deband = core.std.MaskedMerge(ed_deband, ed_denoise, detail_mask[30449:32129])
ed_dering = core.std.MaskedMerge(ed_deband, dering, rings, planes=0)
ending = adptvgrnMod(ed_dering, strength=0.55, cstrength=0, sharp=100, size=1.5, luma_scaling=36, seed=3)

ending = insert_clip(grain, ending, 30449)

final = depth(ending, 10)
#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep07/00002.flac")

final.set_output()

#vspipe --filter-time -c y4m Lupin07.vpy - | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -5 --crqpoffs -5 --qcomp 0.7 --rc-lookahead 120 --crf 15 --no-sao --aq-mode 3 --aq-strength 0.8 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 --zones 2446,4530,b=0.9/11970,13199,q=5 - -o /mnt/HDD1/lupin/ep07/final.hevc
