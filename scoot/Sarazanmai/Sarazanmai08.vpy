import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core

banded = [(1217, 1336), (2366, 2422), (7546, 7619), (11017, 11213), (18270, 19745), (21666, 22247), (22743, 22856), (24678, 25277), (26478, 32572), (32656, None)]

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep08/00001.m2ts')[25:32825]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')

##FILTERING
aa1 = lvf.aa.transpose_aa(src, eedi3=True)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=2)

dm = lvf.mask.detail_mask(aa, brz_a=900, brz_b=3<<8)

dblY = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
dbl = core.resize.Spline36(dblY, 1920, 1080, format=vs.YUV444P16)
dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=1, channels='UV', device_type='gpu')
dbl = core.resize.Spline36(dbl, 1920, 1080, format=vs.YUV420P16)
db = debandshit.dumb3kdb(dbl, radius=18, threshold=[30, 36, 36], grain=0)

db = core.std.MaskedMerge(db, aa, dm)
dblY = core.std.MaskedMerge(dblY, aa, dm)
db = lvf.rfs(dblY, db, banded)

#ED stuff
eds = aa[29299:31577]
edbl = db[29299:31577]

dm2 = lvf.mask.detail_mask(edbl, brz_a=5<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edbl, eds, dm2)

edg = adptvgrnMod(edms, strength=0.35, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, cstrength=0.15, sharp=65, size=1.1, luma_scaling=18)
grn = adptvgrnMod(grn, strength=0.25, sharp=55, size=1, luma_scaling=8, static=True)

edf = insert_clip(grn, edg, 29299)

#core.std.Interleave([src, edf], mismatch=True).set_output()
#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep08/00001.m2ts'), trims=(25, 32825), audio_file='/home/c0mp/sarazanmai/ep08/00001.flac')

final = depth(edf, 10)
final.set_output()
