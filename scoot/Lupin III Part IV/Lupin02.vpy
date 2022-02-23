import vapoursynth as vs
import vardefunc as vdf
import havsfunc as haf
import lvsfunc as lvf
import debandshit
import vsmask

from vsutil import depth, get_y, iterate, insert_clip, plane, join, split
from adptvgrnMod import adptvgrnMod
from zzfunc.std import LevelsM
from acsuite import eztrim

core = vs.core
filepath = "/mnt/HDD1/lupin/"
sp = lvf.comparison.stack_planes

denoise_exclude = [(1080, 1221), (1492, 1548), (1657, 1779), (2038, 3956),
				(4467, 4706), (4815, 5126), (30449, 32127)]

stronger_deband_ranges = [(7850, 9810), (9841, 10208), (17584, 17880), (18748, 18981),
                    (19387, 19467), (20471, 20852), (21433, 21548), (21834, 24618), 
                    (25633, 25668), (28099, 28773), (29425, 29508), (32200, 32289), (32459, 32513), (32650, 32715)] 

def texture_mask(clip, radius, points=[x * 256 for x in (1.75, 2.5, 5, 10)]):
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

    mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
    return mask

# sources
gt = core.lsmas.LWLibavSource(f"{filepath}ep02/00003.m2ts")
src = core.std.Crop(gt, top=2)[:32848]
src = depth(src, 16)

ed = core.lsmas.LWLibavSource(f"{filepath}/ED.mkv")

# denoise
uv_denoise = lvf.rfs(core.knlm.KNLMeansCL(src, d=3, a=4, h=0.5, channels='UV', device_type='gpu'), src, denoise_exclude[:2] + denoise_exclude[-3:])
denoise = haf.SMDegrain(src, tr=6, thSAD=150)
denoise = lvf.rfs(denoise, uv_denoise, denoise_exclude)

crepair = lvf.deblock.vsdpir(denoise, task="deblock", matrix=1, cuda=True)
crepair = vdf.merge_chroma(denoise, crepair)
crepair = lvf.rfs(crepair, denoise, denoise_exclude + [(0, 797)]) 

# deband
deband = debandshit.dumb3kdb(crepair, radius=16, threshold=[36, 32, 32], grain=0)
deband = lvf.rfs(deband, denoise, denoise_exclude)

strong_deband = debandshit.dumb3kdb(denoise, radius=26, threshold=[64, 56, 56])
strong_deband = adptvgrnMod(strong_deband, strength=0.15)
deband = lvf.rfs(deband, strong_deband, stronger_deband_ranges)

detail_mask = lvf.mask.detail_mask(denoise, rad=3, brz_a=0.008, brz_b=0.05)
texture = texture_mask(denoise, radius=3)

deband = core.std.MaskedMerge(deband, crepair, detail_mask)
deband = core.std.MaskedMerge(deband, crepair, texture)

grain = adptvgrnMod(deband, strength=0.35, cstrength=0.1, size=1.25, sharp=75, luma_scaling=8)

final = depth(grain, 10)
final = insert_clip(final, ed, 30449)

#eztrim(gt, trims=(None, 32848), audio_file=f"{filepath}ep01/00002.opus")
final.set_output()
src.set_output(1)

sp(src).set_output(3)
sp(final).set_output(4)

detail_mask.set_output(5)
#less br 59->204
#vspipe --filter-time Lupin02.vpy - -c y4m | x265 --no-strong-intra-smoothing --y4m --preset veryslow --bframes 16 --no-open-gop --no-cutree --rd 3 --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -1 --crqpoffs -1 --qcomp 0.7 --rc-lookahead 120 --crf 15.8 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2.1 --ipratio 1.25 --pbratio 1.15 --zones 59,204,b=0.9/2038,3956,b=0.9/30449,32127,b=0.9 - -o /mnt/HDD1/lupin/ep02/final.hevc
