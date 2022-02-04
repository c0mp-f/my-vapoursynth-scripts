import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as haf
import vardefunc as vdf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import depth, insert_clip, split, join, get_y

##SETUP
core = vs.core

banded = [(1, 1204), (3936, 4115), (4608, 5823), (6052, 7059), (7384, 7563), (7612, 7881), (8444, 8878), (9143, 9954), (10211, 10273), (11213, 11893), (12080, 12242), (13087, 13832), (14085, 15775), (16112, 16186), (16427, 16609), (17411, 28943), (28992, None)] # This episode is fucking plagued with banding and blocking; scenefiltering has become impractical, I quit.

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep09/00002.m2ts')[:32798]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')


##FILTERING
aa1 = lvf.aa.transpose_aa(src, eedi3=True)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=3)

dblY = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
#dbl = core.resize.Spline36(dblY, 1920, 1080, format=vs.YUV444P16)
#dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=0.45, channels='UV', device_type='gpu')
#dbl = core.resize.Spline36(dbl, 1920, 1080, format=vs.YUV420P16)

db = debandshit.dumb3kdb(aa, radius=20, threshold=[12, 10, 10], grain=0)

dm = vdf.mask.detail_mask(dblY, brz_mm=2<<8, brz_ed=3<<8)

db = lvf.rfs(aa, db, banded)
db = core.std.MaskedMerge(db, aa, dm)

lines = core.imwri.Read('customask.png')
lines = core.resize.Point(lines, format=vs.GRAY16, matrix=1)
lines = lines * (17894 - 17847)
detail = core.std.MaskedMerge(db[17847:17894], aa[17847:17894], lines)
db = insert_clip(db, detail, 17847)

#ED stuff
eds = aa[25966:28243]

edbl = core.resize.Spline36(eds, 1920, 1080, format=vs.YUV444P16)
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, h=0.25, channels='Y', device_type='gpu')
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, h=0.9, channels='UV', device_type='gpu')
edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV420P16)

edb = debandshit.dumb3kdb(edbl, threshold=[0, 32, 32], grain=0)

dm1 = lvf.mask.detail_mask(edbl, brz_a=2<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edbl, eds, dm1)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.17, sharp=55, size=1.05, luma_scaling=8, static=True)
grn = adptvgrnMod(grn, strength=0.2, cstrength=0.18, sharp=65, size=1.1, luma_scaling=18)

edf = insert_clip(grn, edg, 25966)

#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep09/00002.m2ts'), trims=(None, 32798), audio_file='/home/c0mp/sarazanmai/ep09/00002.flac')
#core.std.Interleave([src, edf], mismatch=True).set_output()
#core.std.Interleave([src[17847:17895], detail], mismatch=True).set_output()

final = depth(edf, 10)
final.set_output()
