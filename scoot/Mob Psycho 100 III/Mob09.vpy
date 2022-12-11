import vapoursynth as vs 
import vardefunc as vdf
import havsfunc as haf
import awsmfunc as awf
import lvsfunc as lvf
import vstools as vst
import vsmask

from vsaa import Eedi3, Eedi3SR, upscaled_sraa, clamp_aa, transpose_aa, Nnedi3SR
from vsutil import depth, insert_clip, get_y, split, join
from vsscale import descale_detail_mask, ssim_downsample
from lvsfunc.comparison import stack_planes as sp
from debandshit import placebo_deband, dumb3kdb
from vardefunc import Graigasm, AddGrain
from vsaa import transpose_aa, Eedi3
from EoEfunc.denoise import BM3D
from kagefunc import squaremask
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

    deblock = DPIR(rgbs, strength=11, task="deblock")
    deblock2 = DPIR(rgbs, strength=25, task="deblock")

    deblock = vst.rfs(deblock, deblock2, [(1347, 1483), (1759, 2031)])

    deblock = core.resize.Bicubic(deblock, format=vs.YUV420P16, matrix=1)
    deblock = vst.rfs(denoise, deblock, [(216, 251), (320, 594), (730, 764), (867, 2155)])

    crop = deblock.std.Crop(0, 0, 138, 138)
    bb = awf.bbmod(crop, top=1, bottom=1, cBottom=1, cTop=1)
    letterbox = bb.std.AddBorders(top=138, bottom=138)
    
    op = vst.rfs(deblock, letterbox, (354, 594))

    return op

OP = [(936, 3092)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [(31529, 33686)]

regrain_ranges = [
    (opstart+216, opstart+248), (opstart+755, opstart+764), (opstart+850, opstart+866)
]
dpir_ranges = [
    (7599, 7640), (29877, 29917)
] + ED

stronger_deband_ranges = [(29877, 29917)]

NO_DESCALE = OP
NO_EDGEFIX = OP

# setup
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 - 09 (1080p) [F34EE0A7].mkv")
funi = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[Yameii] Mob Psycho 100 III - 09 [English Dub] [WEB-DL 1080p] [E39B1779].mkv")[167:]

src = depth(gt, 16)

preden = core.dfttest.DFTTest(src, sigma=5)

detail_mask = lvf.mask.detail_mask(preden, rad=2, brz_a=0.015, brz_b=0.015)

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

# denoising
rgbs = Bicubic().resample(rescale, vs.RGBS, matrix_in=1)
deblock = DPIR(rgbs, strength=4, task="deblock")
deblock = Bicubic().resample(deblock, vs.YUV420P16, matrix=1)

deblock = vst.rfs(rescale, deblock, dpir_ranges)

cden = ccd(deblock, 5)
denoise = haf.SMDegrain(cden, tr=6, thSAD=150, prefilter=3, RefineMotion=True, plane=0, chroma=False)
denoise = BM3D(deblock, ref=denoise, radius=2, sigma=1.2, CUDA=False)

denoise = vst.rfs(denoise, deblock, OP)

# deband settings for the episode
deband = placebo_deband(denoise, 20, [1.7, 1.3], 1, 0)

stronger_deband = placebo_deband(denoise, 28, [2.3, 1.7], 2, 0)

# deband settings for the OP
averaged_deband = core.average.Mean([
    denoise,
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
deband = core.std.MaskedMerge(deband, denoise, detail_mask)

# graining
grain = Graigasm(
    thrs=[x << 8 for x in (50, 150)], 
    strengths=[(0.25, 0), (0.25, 0)], 
    sizes=[1.3, 1.3], sharps=[65, 60],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

regrained = core.noise.Add(grain, type=0, var=7, seed=69)
grain = vst.rfs(grain, regrained, regrain_ranges)

final = depth(grain, 10)

from stgfunc import output as o

final.set_output()

o(gt)
o(sp(gt))
o(sp(final))

o(detail_mask)

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/Mob\ Psycho\ 100\ III/Mob09.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-3 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.73 --rc-lookahead 120 --crf 13 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2 --ipratio 1.3 --pbratio 1.2 - -o /mnt/HDD1/mob/ep09/final.hevc
