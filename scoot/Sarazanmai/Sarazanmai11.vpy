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

deband = [(3193, 4116), (5347, 5382), (5674, 6237), (6403, 6439), (6821, 6993), (7507, 9404), (13917, 17191), (17975, 18118), (18435, 18800), (19313, 22263)]
aa_chroma_range = [(5347, 5382), (8518, 8601), (14394, 17484), (25793, 26434)]

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep11/00002.m2ts')[:32798]
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

db = debandshit.dumb3kdb(dbl, radius=20, threshold=20, grain=0)

db = core.std.MaskedMerge(db, aa, dm)
db = lvf.rfs(dbl, db, deband)

grn = adptvgrnMod(db, strength=0.25, sharp=55, size=1.1, luma_scaling=8, static=True)
grn = adptvgrnMod(grn, strength=0.25, cstrength=0.15, sharp=60, size=1.1, luma_scaling=18)

#core.std.Interleave([src, edf], mismatch=True).set_output()
eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep11/00002.m2ts'), trims=(None, 32798), audio_file='/home/c0mp/sarazanmai/ep11/00002.flac')

final = depth(grn, 10)
final.set_output()
