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

deband = [(168, 215), (3486, 3995), (4419, 4478), (7344, 7607), (8196, 8664), (8695, 9546), (10250, 10333), (10781, 11377), (11813, 13288), (14401, 20111), (21276, 23362), (23549, 23848), (23939, 24040), (24218, 24610), (24707, 30807)]
aa_chroma_range = [(15217, 17254), (18777, 18959), (25001, 25066)]

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep10/00001.m2ts')[24:32822]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')


##FILTERING
aa_planes = [lvf.aa.transpose_aa(p, eedi3=True) for p in split(src)]
aa_chroma = join(aa_planes)
aa_chroma = core.resize.Spline36(aa_chroma, 1920, 1080, format=vs.YUV420P16)

aa1 = lvf.aa.transpose_aa(src, eedi3=True)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=2)

specialAAm = core.tcanny.TCanny(aa_chroma, sigma=1.2, planes=0).std.Maximum().std.Inflate()
specialAA = core.std.MaskedMerge(aa, aa_chroma, specialAAm)
aa = lvf.rfs(aa, specialAA, aa_chroma_range)

dm = lvf.mask.detail_mask(aa, brz_a=900, brz_b=3<<8)

dbl = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')

db = debandshit.dumb3kdb(dbl, radius=20, threshold=[20, 20, 20], grain=0)

db = core.std.MaskedMerge(db, aa, dm)
db = lvf.rfs(dbl, db, deband)

#ED stuff
eds = aa[28531:30807]

edbl = core.resize.Spline36(eds, 1920, 1080, format=vs.YUV444P16)
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, h=0.9, channels='UV', device_type='gpu')
edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV420P16)

edb = debandshit.dumb3kdb(edbl, threshold=[0, 26, 26], grain=0)

dm2 = lvf.mask.detail_mask(edbl, brz_a=5<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edb, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, sharp=55, size=1.1, luma_scaling=8, static=True)
grn = adptvgrnMod(grn, strength=0.25, cstrength=0.2, sharp=75, size=1.1, luma_scaling=18)

edf = insert_clip(grn, edg, 28531)

#core.std.Interleave([src, edf], mismatch=True).set_output()
#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep10/00001.m2ts'), trims=(24, 32822), audio_file='/home/c0mp/sarazanmai/ep10/00001.flac')

final = depth(edf, 10)
final.set_output()
