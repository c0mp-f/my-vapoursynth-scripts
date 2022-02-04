import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import debandshit
from adptvgrnMod import adptvgrnMod
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core
#core.max_cache_size = 28000
#core.num_threads = 14

specialAArange = [(6036, 6204), (19369, 22695)]

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep04/00001.m2ts')[24:32825]
src = depth(src, 16)
src = core.resize.Point(src, matrix=1, matrix_in=1, range=0, range_in=1, primaries=1, primaries_in=1, transfer=1, transfer_in=1, dither_type='error_diffusion')

##FILTERING

#descale = kgf.inverse_scale(src, height=860, kernel='bicubic', b=0, c=1/2)
#descale = depth(descale, 16)
#rescale = core.resize.Bicubic(descale, 1920, 1080, filter_param_a=0, filter_param_b=1/2, format=vs.YUV420P16)
#rescale = core.resize.Spline16(descale, 1920, 1080, format=vs.YUV420P16)
#rescale = core.resize.Lanczos(descale, 1920, 1080, filter_param_a=2, format=vs.YUV420P16)

aa_planes = [lvf.aa.transpose_aa(p, eedi3=True) for p in split(src)]
aachroma = join(aa_planes)
aachroma = core.resize.Spline36(aachroma, 1920, 1080, format=vs.YUV420P16)

aa1 = lvf.aa.transpose_aa(src, eedi3=True)

aa2 = lvf.sraa(src)
aa2 = core.resize.Spline36(aa2, 1920, 1080, format=vs.YUV420P16)

aa = lvf.aa.clamp_aa(src=src, weak=aa2, strong=aa1, strength=2)

specialAAm = get_y(core.tcanny.TCanny(aachroma, sigma=1.2)).std.Maximum().std.Inflate()
specialAA = core.std.MaskedMerge(aa, aachroma, specialAAm)

aa = lvf.rfs(aa, specialAA, specialAArange)

dbl = core.knlm.KNLMeansCL(aa, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
db = debandshit.dumb3kdb(dbl, radius=18, threshold=[60, 32, 32], grain=0)
dm = lvf.mask.detail_mask(aa, brz_a=900, brz_b=1000)
dbl = core.std.MaskedMerge(dbl, aa, dm)
db = core.std.MaskedMerge(db, aa, dm)

#ED stuff
eds = src[29684:31963]
edbl = dbl[29684:31963]
edbl = core.resize.Spline36(edbl, 1920, 1080, format=vs.YUV444P16)
edbl = core.knlm.KNLMeansCL(edbl, d=3, a=4, channels='UV', device_type='gpu')

dm2 = lvf.mask.detail_mask(edbl, brz_a=5<<8, brz_b=8<<8)
edb = core.resize.Spline36(edbl, format=vs.YUV420P16)
edb = debandshit.dumb3kdb(edb, preset='medium,nograin,chroma')
edms = core.std.MaskedMerge(edb, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, sharp=55, size=1.05, luma_scaling=14, static=True)
grn = adptvgrnMod(grn, strength=0.25, cstrength=0.1, sharp=60, size=1.1, luma_scaling=12)

edf = insert_clip(grn, edg, 29684)

#core.std.Interleave([rescale, src], mismatch=True).set_output()
#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep04/00001.m2ts'), trims=(24, 32825), audio_file='/home/c0mp/sarazanmai/ep04/00001.flac')
final = depth(edf, 10)
final.set_output()
