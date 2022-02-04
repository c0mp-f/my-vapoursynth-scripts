import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep06/00001.m2ts')[24:32824]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')


##FILTERING
aa_planes = [lvf.aa.transpose_aa(p, eedi3=True) for p in split(src)]
aachroma = join(aa_planes)
aachroma = core.resize.Spline36(aachroma, 1920, 1080, format=vs.YUV420P16)

aa1 = lvf.aa.transpose_aa(src, eedi3=True)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=2)

specialAAm = get_y(core.tcanny.TCanny(aachroma, sigma=1.2)).std.Maximum().std.Inflate()
specialAA = core.std.MaskedMerge(aa[27878:28064], aachroma[27878:28064], specialAAm)
aa = insert_clip(aa, specialAA, 27878)

dm = lvf.mask.detail_mask(aa, brz_a=3<<8, brz_b=3<<8)

dbl = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
db = debandshit.dumb3kdb(dbl, threshold=[40, 44, 44], grain=0)

db = core.std.MaskedMerge(db, aa, dm)

#ED stuff
eds = aa[29659:31938]
edbl = db[29659:31938]
edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV444P16)
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, h=1, channels='UV', device_type='gpu')

dm2 = lvf.mask.detail_mask(edbl, brz_a=5<<8, brz_b=8<<8)
edb = core.resize.Spline36(edbl, format=vs.YUV420P16)
edms = core.std.MaskedMerge(edb, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, sharp=55, size=1.05, luma_scaling=10)
grn = adptvgrnMod(grn, strength=0.28, sharp=65, size=1.1, luma_scaling=12)

edf = insert_clip(grn, edg, 29659)

#core.std.Interleave([src, edf], mismatch=True).set_output()
#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep06/00001.m2ts'), trims=(24, 32824), audio_file='/home/c0mp/sarazanmai/ep06/00001.flac')

final = depth(edf, 10)
final.set_output()
