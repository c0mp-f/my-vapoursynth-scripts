import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as haf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep07/00002.m2ts')[:32798]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')


##FILTERING
aa_planes = [lvf.aa.transpose_aa(p, eedi3=True) for p in split(src)]
aachroma = join(aa_planes)
aachroma = core.resize.Spline36(aachroma, 1920, 1080, format=vs.YUV420P16)

aa1 = lvf.aa.transpose_aa(src, eedi3=True)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=3)

specialAAm = get_y(core.tcanny.TCanny(aachroma[23757:26480], sigma=1.2)).std.Maximum().std.Inflate()
specialAA = core.std.MaskedMerge(aa[23757:26480], aachroma[23757:26480], specialAAm)
aa = insert_clip(aa, specialAA, 23757)

dm = lvf.mask.detail_mask(aa, brz_a=3<<8, brz_b=5<<8)

dbl = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
dbl = core.resize.Spline36(dbl, 1920, 1080, format=vs.YUV444P16)
dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=1, channels='UV', device_type='gpu')
dbl = core.resize.Spline36(dbl, 1920, 1080, format=vs.YUV420P16)

db = debandshit.dumb3kdb(dbl, radius=18, threshold=[36, 40, 40], grain=0)

db = core.std.MaskedMerge(db, aa, dm)

#ED stuff
eds = aa[28459:30736]
edbl = dbl[28459:30736]

dm2 = lvf.mask.detail_mask(edbl, brz_a=5<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edbl, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.2, sharp=55, size=1.1, luma_scaling=8, static=True)
grn = adptvgrnMod(grn, strength=0.25, cstrength=0.2, sharp=75, size=1.1, luma_scaling=18)

edf = insert_clip(grn, edg, 28459)

#core.std.Interleave([src, edf], mismatch=True).set_output()
#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep07/00002.m2ts'), trims=(None, 32798), audio_file='/home/c0mp/sarazanmai/ep07/00002.flac')

final = depth(edf, 10)
final.set_output()
