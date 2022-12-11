import vapoursynth as vs 
import vardefunc as vdf
import havsfunc as haf
import awsmfunc as awf
import lvsfunc as lvf
import vstools as vst
import vsmask

from vsaa import Eedi3, Eedi3SR, upscaled_sraa, clamp_aa, transpose_aa, Nnedi3SR, SangNomSR
from vsutil import depth, insert_clip, get_y, split, join
from vsscale import descale_detail_mask, ssim_downsample
from lvsfunc.comparison import stack_planes as sp
from debandshit import placebo_deband, dumb3kdb
from vardefunc import Graigasm, AddGrain
from vsaa import transpose_aa, Eedi3
from EoEfunc.denoise import BM3D
from kagefunc import squaremask
from filter import texture_mask
from vskernels import Bicubic, BSpline
from rekt import rektlvls
from vsdpir import DPIR
from ccd import ccd

core = vs.core

def OP_filtering(clip: vs.VideoNode) -> vs.VideoNode:
    
    denoise = haf.SMDegrain(clip, tr=6, thSAD=260, RefineMotion=True)
    denoise = depth(denoise, 32)
    denoise = core.wnnm.WNNM(denoise, sigma=[1, 1, 1])
    denoise = BM3D(clip, ref=denoise, sigma=1.5, radius=2, CUDA=False)
    rgbs = core.resize.Bicubic(denoise, format=vs.RGBS, matrix_in=1)

    deblock = DPIR(rgbs, strength=7, task="deblock")
    deblock2 = DPIR(rgbs, strength=25, task="deblock")

    deblock = vst.rfs(deblock, deblock2, [(1347, 1483), (1759, 2031)])

    deblock = core.resize.Bicubic(deblock, format=vs.YUV420P16, matrix=1)
    deblock = vst.rfs(denoise, deblock, [(867, 2155)])

    crop = deblock.std.Crop(0, 0, 138, 138)
    bb = awf.bbmod(crop, top=1, bottom=1, cBottom=1, cTop=1)
    letterbox = bb.std.AddBorders(top=138, bottom=138)
    
    op = vst.rfs(deblock, letterbox, (354, 594))

    return op

OP = [(2949, 5106)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [(31528, 33685)]

regrain_ranges = [
    (opstart+216, opstart+248), (opstart+755, opstart+764), (opstart+850, opstart+866)
]
dpir_ranges = [
    (11554, 11838), (15769, 15889), (18815, 19909), (20253, 20430), 
    (20603, 20679), (22995, 23460)
] + ED

stronger_deband_ranges = [
    (158, 175), (5299, 5400), (6304, 6327), 
    (9708, 9790), (14749, 14931)
]

holyfuckitsawful = [
    (11554, 11726)
]

# 15863 sources mix
# 17062, 17183
# 22995, 23460

NO_DESCALE = OP + ED
NO_EDGEFIX = OP

# setup
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 - 10v2 (1080p) [3B717070].mkv")
bg = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[NC-Raws] 路人超能100 III - 10 (B-Global 3840x2160 HEVC AAC MKV) [3E386811].mkv")

src = depth(gt, 16)
bg = depth(bg, 16)
bg = BSpline().scale(bg, 1920, 1080)

preden = core.dfttest.DFTTest(src, sigma=5)

detail_mask = lvf.mask.detail_mask(preden, 2, brz_a=0.02, brz_b=0.02)
textures = texture_mask(preden, 1, points=[x * 256 for x in (1.25, 2.2, 5, 15)])
textures = vst.rfs(textures, core.std.BlankClip(textures), [(opstart+1347, opstart+1483), (opstart+1759, opstart+2031)]) 

everything_mask = core.std.Expr([detail_mask, textures], "x y max")

# sceneflt. 
op = OP_filtering(src[opstart:opend])
op = insert_clip(src, op, opstart)

# edgefixing
rekt = rektlvls(
    op,
    colnum=[0, 1919], colval=[15, 15],
    rownum=[0, 1079], rowval=[17, 16]
)
rekt = vst.rfs(rekt, op, NO_EDGEFIX)

# descale
descale = Bicubic().descale(rekt, 1280, 720)

rescale = vdf.nnedi3_upscale(descale)
rescale = ssim_downsample(rescale, 1920, 1080)
rescale = vdf.merge_chroma(rescale, rekt)

# the OP seemed like a spooky descale case that i'm better off not touching
rescale = vst.rfs(rescale, rekt, NO_DESCALE)

catrom_rescale = Bicubic().scale(descale, 1920, 1080)
error_mask = descale_detail_mask(rekt, catrom_rescale)

rescale = core.std.MaskedMerge(rescale, rekt, error_mask)

merge = vdf.merge_chroma(core.std.Merge(get_y(rescale), get_y(bg)), src)
merge = vst.rfs(rescale, merge, [(5299, 5400), (11631, 11779), (15762, 15889), (17062, 17183), (21707, 21974), (22995, 23460)])

# denoising
rgbs = Bicubic().resample(merge, vs.RGBS, matrix_in=1)
deblock = DPIR(rgbs, strength=7, task="deblock")
deblock = Bicubic().resample(deblock, vs.YUV420P16, matrix=1)

deblock = vst.rfs(merge, deblock, dpir_ranges)

cden = ccd(deblock, 5)
denoise = haf.SMDegrain(cden, tr=6, thSAD=150, prefilter=3, RefineMotion=True, plane=0, chroma=False)
denoise = BM3D(deblock, ref=denoise, radius=2, sigma=1.2, CUDA=False)

denoise = vst.rfs(denoise, deblock, OP)

eedi = Eedi3()
eedi.alpha = 0.8
eedi.beta = 0
eedi.gamma = 0

aa = upscaled_sraa(denoise, aafunc=eedi)

linemask = vsmask.edge.ExKirsch().edgemask(get_y(preden))
linemask = linemask.std.Binarize(200<<8).std.Maximum()

aa = core.std.MaskedMerge(denoise, aa, linemask)
aa = vst.rfs(denoise, aa, holyfuckitsawful)

# deband settings for the episode
deband = placebo_deband(aa, 20, [1.7, 1.3], 1, 0)

stronger_deband = placebo_deband(aa, 28, [2.3, 1.7], 2, 0)

# deband settings for the OP
averaged_deband = core.average.Mean([
    aa,
    placebo_deband(denoise, 4, 1.8, 2, 0),
    placebo_deband(denoise, 6, 2.4, 2, 0),
    placebo_deband(denoise, 10, 2, 2, 0),
    placebo_deband(denoise, 12, 2, 1, 0)
]) 

op_weak_deband = core.neo_f3kdb.Deband(averaged_deband, y2=2, cb2=2, cr2=2, range=30, grainy=0, grainc=0)
op_stronger_deband = placebo_deband(averaged_deband, 32, [1.8, 2], 1, 0)
op_deband = vst.rfs(op_stronger_deband, op_weak_deband, (opstart+1278, opstart+1346))

deband = vst.rfs(deband, op_deband, OP)
deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = core.std.MaskedMerge(deband, denoise, everything_mask)

# graining
grain = Graigasm(
    thrs=[x << 8 for x in (50, 75, 150)], 
    strengths=[(0.25, 0), (0.2, 0), (0.25, 0)], 
    sizes=[1.3, 1.3, 1.3], sharps=[65, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

regrained = core.noise.Add(grain, type=0, var=7, seed=69)
grain = vst.rfs(grain, regrained, regrain_ranges)

final = depth(grain, 10)

lossless = core.lsmas.LWLibavSource("/mnt/HDD1/mob/ep10/lossless.mkv")
lossless = Graigasm(
    thrs=[x << 8 for x in (50, 75, 125)], 
    strengths=[(0.25, 0), (0.2, 0), (0.25, 0)], 
    sizes=[1.3, 1.3, 1.3], sharps=[65, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(lossless)
regrained = core.noise.Add(grain, type=0, var=7, seed=69)
lossless = vst.rfs(lossless, regrained, regrain_ranges)
lossless = depth(lossless, 10)
final = vst.rfs(lossless, final, [(5299, 5400), (11631, 11779), (21707, 21974), (opstart+1347, opstart+1483), (opstart+1759, opstart+2031)] + holyfuckitsawful)

from stgfunc import output as o

final.set_output()

o(gt)
o(denoise)
o(aa)
o(linemask)

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/Mob\ Psycho\ 100\ III/Mob10.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-3 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.73 --rc-lookahead 120 --crf 14 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2 --ipratio 1.3 --pbratio 1.2 - -o /mnt/HDD1/mob/ep10/final.hevc
