import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core
#core.max_cache_size = 26700 
#core.num_threads = 14

banded = [(1, 463), (12154, 15079), (16010, 18079), (18296, 18530), (18875, 19030), (19878, 23592), (23883, 26942), (27099, 28769), (31048, 32795)]
aarange = [(3657, 6250), (8261, 8690), (10440, 14563), (16913, 23046), (24756, 26153), (27123, 28770)]
#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep03/00002.m2ts')[2:32798]
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
aa = lvf.rfs(src, aa, aarange)

specialAA = aachroma[19880:23047]
specialAAm = get_y(core.tcanny.TCanny(specialAA, sigma=1.2)).std.Maximum().std.Inflate()
specialAA = core.std.MaskedMerge(aa[19880:23047], specialAA, specialAAm)
aa = insert_clip(aa, specialAA, 19880)

dbl = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
db = debandshit.dumb3kdb(dbl, threshold=64, grain=0)
dm = lvf.mask.detail_mask(aa, brz_a=3<<8, brz_b=5<<8)
dbl = core.std.MaskedMerge(dbl, aa, dm)
db = core.std.MaskedMerge(db, aa, dm)

db = lvf.rfs(dbl, db, banded)

strongdb = aa[4900:4946]
strongdb = debandshit.dumb3kdb(strongdb, threshold=80, grain=0)
strongdb = core.std.MaskedMerge(strongdb, src[4900:4946], dm[4900:4946])
db = insert_clip(db, strongdb, 4900)

#ED stuff
eds = aa[28770:31047]
edbl = dbl[28770:31047]
edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV444P16)
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, channels='UV', device_type='gpu')

dm2 = lvf.mask.detail_mask(edbl, brz_a=5<<8, brz_b=8<<8)
edb = core.resize.Spline36(edbl, format=vs.YUV420P16)
edb = debandshit.dumb3kdb(edb, preset='medium,nograin,chroma')
edms = core.std.MaskedMerge(edb, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, sharp=55, size=1.05, luma_scaling=10, static=True)
grn = adptvgrnMod(grn, strength=0.3, cstrength=0.1, sharp=70, size=1.1, luma_scaling=14)

edf = insert_clip(grn, edg, 28770)

#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep03/00002.m2ts'), trims=(2, 32798), audio_file='/home/c0mp/sarazanmai/ep03/00002.opus')
final = depth(edf, 10)
final.set_output()
