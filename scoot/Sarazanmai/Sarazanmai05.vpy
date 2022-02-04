import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import debandshit
from adptvgrnMod import adptvgrnMod
from awsmfunc import fixlvls
from acsuite import eztrim
from vsutil import *

##SETUP
core = vs.core
#core.max_cache_size = 28000
#core.num_threads = 14

specialAArange = [(20159, 22936), (24337, 24481)]

#import
src = core.lsmas.LWLibavSource('/home/c0mp/sarazanmai/ep05/00002.m2ts')[:32800]
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

specialAAm = get_y(core.tcanny.TCanny(aachroma, sigma=1.2)).std.Maximum().std.Inflate()
specialAA = core.std.MaskedMerge(aa, aachroma, specialAAm)

aa = lvf.rfs(aa, specialAA, specialAArange)

badlines = src[12008:12020]
lines = lvf.sraa(badlines, rfactor=1.2)
lines = core.std.MaskedMerge(badlines, lines, specialAAm[12008:12020])
aa = insert_clip(aa, lines, 12008)

dm = lvf.mask.detail_mask(aa, brz_a=3<<8, brz_b=3<<8)

dbl = core.resize.Spline36(aa, 1920, 1080, format=vs.YUV444P16)
dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=0.3, channels='Y', device_type='gpu')
dbl = core.knlm.KNLMeansCL(dbl, d=3, a=4, h=1, channels='UV', device_type='gpu')
dbl = core.resize.Spline36(dbl, 1920, 1080, format=vs.YUV420P16)

db = debandshit.dumb3kdb(dbl, threshold=[30, 34, 34], grain=0)

db = core.std.MaskedMerge(db, aa, dm)

#ED stuff
eds = aa[29635:31914]
edbl = db[29635:31914]

dm2 = lvf.mask.detail_mask(edbl, brz_a=4<<8, brz_b=8<<8)
edms = core.std.MaskedMerge(edbl, eds, dm2)

edg = adptvgrnMod(edms, strength=0.3, cstrength=0.15, size=1.1, sharp=60, luma_scaling=72)

grn = adptvgrnMod(db, strength=0.25, sharp=50, size=1.05, luma_scaling=8)
grn = adptvgrnMod(grn, strength=0.27, sharp=60, size=1.1, luma_scaling=14)

edf = insert_clip(grn, edg, 29635)

#core.std.Interleave([src, edf], mismatch=True).set_output()
#eztrim(lvf.misc.source('/home/c0mp/sarazanmai/ep05/00002.m2ts'), trims=(None, 32800), audio_file='/home/c0mp/sarazanmai/ep05/00002.flac')

final = depth(edf, 10)
final.set_output()