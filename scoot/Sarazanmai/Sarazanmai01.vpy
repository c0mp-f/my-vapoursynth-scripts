import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import debandshit
from adptvgrnMod import adptvgrnMod
from vsutil import *

##SETUP
core = vs.core
#core.max_cache_size = 27000 
#core.num_threads = 14

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep01/00001.m2ts')[24:32824]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')

##FILTERING
aa_planes = [lvf.aa.transpose_aa(p, eedi3=True) for p in split(src)]
aa1 = join(aa_planes)
aa1 = core.resize.Spline36(aa1, 1920, 1080, format=vs.YUV420P16)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=2)

specialAAm = get_y(core.tcanny.TCanny(aa1)).std.Maximum().std.Inflate()

aamaskf = core.std.MaskedMerge(aa, aa1, specialAAm, planes=[0, 1, 2])
aa = lvf.rfs(clip_a=aa, clip_b=aamaskf, ranges=(18343, 24343))


dbl = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
db = debandshit.dumb3kdb(dbl, threshold=[32, 32, 32], grain=0)
dm = lvf.mask.detail_mask(aa, brz_a=2<<8, brz_b=3<<8)

db = core.std.MaskedMerge(db, aa, dm)

specialdb = debandshit.dumb3kdb(src[13820:14046], radius=20, grain=0, threshold=[80, 32, 32])
specialdb = core.std.MaskedMerge(specialdb, src[13820:14046], dm[13820:14046])
db = insert_clip(db, specialdb, 13820)

#ED stuff
eds = aa[29275:31553]
edbl = db[29275:31553]

edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV444P16)
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, h=1, channels='UV', device_type='gpu')
edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV420P16)

dm2 = lvf.mask.detail_mask(edbl, brz_a=16<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edbl, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.2, sharp=55, size=1.05, luma_scaling=8, static=True)
grn = adptvgrnMod(grn, strength=0.25, cstrength=0.1, sharp=60, size=1.1, luma_scaling=10)

edf = insert_clip(grn, edg, 29275)
#core.std.Interleave(clips=[src, edf], mismatch=True).set_output()

final = depth(edf, 10)
final.set_output()
