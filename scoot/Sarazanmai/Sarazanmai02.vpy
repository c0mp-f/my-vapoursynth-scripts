import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core
#core.max_cache_size = 27000 
#core.num_threads = 14

banded = [(0, 155), (8014, 8640), (9553, 9660), (10453, 12389), (16317, 19278), (19463, 19594), (20792, 20920), (22217, 25458), (25856, 26137), (26388, 26528), (26718, 26790), (26957, 27563), (27888, 31940), (32018, 32801)]
specialAA = [(22890, 25457)] 
aarange = [(0, 1843), (6118, 10000), (11358, 17564), (17769, 19139), (27402, 29660), (31940, 32826)] # generalized to save computing time

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep02/00001.m2ts')[24:32826]
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

aaM = get_y(core.tcanny.TCanny(aa1, sigma=1.2)).std.Maximum().std.Inflate()

aamaskf = core.std.MaskedMerge(aa, aachroma, aaM)
aa = lvf.rfs(src, aa, aarange)
aa = lvf.rfs(aa, aamaskf, specialAA)

dbl = core.resize.Spline36(aa, 1920, 1080, format=vs.YUV444P16)
dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=1, channels='UV', device_type='gpu')
dbl = core.resize.Spline36(dbl, 1920, 1080, format=vs.YUV420P16)

db = debandshit.dumb3kdb(dbl, preset='medium,nograin,luma')
db = debandshit.dumb3kdb(db, preset='medium,nograin,chroma')

dm = lvf.mask.detail_mask(aa, brz_a=3<<8, brz_b=5<<8)

db = core.std.MaskedMerge(db, aa, dm)

db = lvf.rfs(dbl, db, banded)

#ED stuff
eds = aa[29661:31940]
edb = db[29661:31940]

dm2 = lvf.mask.detail_mask(edb, brz_a=16<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edb, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, sharp=55, size=1.05, luma_scaling=8, static=True)
grn = adptvgrnMod(grn, strength=0.28, cstrength=0.1, sharp=65, size=1.05, luma_scaling=16)

edf = insert_clip(grn, edg, 29661)

#core.std.Interleave([src, edf], mismatch=True).set_output()

#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep02/00001.m2ts'), trims=[(24,32826)], audio_file='/home/c0mp/sarazanmai/ep02/episode_02.opus')
final = depth(edf, 10)
final.set_output()
