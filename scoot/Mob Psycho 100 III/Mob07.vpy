import vapoursynth as vs 
import vardefunc as vdf
import havsfunc as haf
import awsmfunc as awf
import lvsfunc as lvf
import vsmask

from vsaa import Eedi3, Eedi3SR, upscaled_sraa, clamp_aa, transpose_aa, Nnedi3SR
from vsutil import depth, insert_clip, get_y, split, join
from vsscale import descale_detail_mask, ssim_downsample
from lvsfunc.comparison import stack_planes as sp
from debandshit import placebo_deband, dumb3kdb
from vardefunc import Graigasm, AddGrain
from vsaa import transpose_aa, Eedi3
from EoEfunc.denoise import BM3D
from filter import texture_mask
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

    deblock = lvf.rfs(deblock, deblock2, [(1347, 1483), (1759, 2031)])

    deblock = core.resize.Bicubic(deblock, format=vs.YUV420P16, matrix=1)
    deblock = lvf.rfs(denoise, deblock, [(216, 251), (320, 594), (730, 764), (867, 2155)])

    crop = deblock.std.Crop(0, 0, 138, 138)
    bb = awf.bbmod(crop, top=1, bottom=1, cBottom=1, cTop=1)
    letterbox = bb.std.AddBorders(top=138, bottom=138)
    
    op = lvf.rfs(deblock, letterbox, (354, 594))

    return op

OP = [(4245, 6401)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [(31527, 33684)]

regrain_ranges = [
    (opstart+216, opstart+248), (opstart+755, opstart+764), (opstart+850, opstart+866)
]
dpir_ranges = [
    (12407, 12535), (23649, 23913), (26762, 26844), (29527, 29597)
] + ED
stronger_deband_ranges = [
    (12108, 12406), (12536, 13216), (15755, 16151), (16276, 16748),
    (27909, 27965), (28664, 28842), (29527, 29597), (33845, 33902)
]

NO_DESCALE = OP + ED
NO_EDGEFIX = OP

# setup
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 - 07 (1080p) [34EEB900].mkv")

bili = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[NC-Raws] 路人超能100 III - 07 (B-Global 3840x2160 HEVC AAC MKV) [F8D4398A].mkv")

bili_untouched = depth(bili, 16)

bili = BSpline().scale(bili_untouched, 1920, 1080)
bili = vdf.merge_chroma(bili, bili_untouched)
bili = Bicubic().resample(bili, vs.YUV420P16)

src = depth(gt, 16)

preden = core.dfttest.DFTTest(src, sigma=5)

detail_mask = lvf.mask.detail_mask(preden, rad=2, brz_a=0.015, brz_b=0.015)
textures = texture_mask(preden, radius=2, points=[x * 256 for x in (1, 2, 5, 15)])
textures = lvf.rfs(textures, core.std.BlankClip(textures), [(opstart+1347, opstart+1483), (opstart+1759, opstart+2031)]) 

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
rekt = lvf.rfs(rekt, op, NO_EDGEFIX)

# descale
descale = Bicubic().descale(rekt, 1280, 720)

rescale = vdf.nnedi3_upscale(descale)
rescale = ssim_downsample(rescale, 1920, 1080)
rescale = vdf.merge_chroma(rescale, rekt)

# the OP seemed like a spooky descale case that i'm better off not touching
rescale = lvf.rfs(rescale, rekt, NO_DESCALE)

catrom_rescale = Bicubic().scale(descale, 1920, 1080)
error_mask = descale_detail_mask(rekt, catrom_rescale, threshold=7*257)

rescale = core.std.MaskedMerge(rescale, rekt, error_mask)

average = core.average.Mean([rescale, bili])
scenef = lvf.rfs(rescale, average, (26762, 26844))
scenef = Bicubic().resample(scenef, vs.YUV420P16)

# denoising
rgbs = Bicubic().resample(scenef, vs.RGBS, matrix_in=1)
deblock = DPIR(rgbs, strength=6, task="deblock")
deblock = Bicubic().resample(deblock, vs.YUV420P16, matrix=1)

deblock = lvf.rfs(scenef, deblock, dpir_ranges)

cden = depth(core.wnnm.WNNM(depth(deblock, 32), sigma=[0, 0.85, 0.85], radius=2), 16)
denoise = haf.SMDegrain(cden, tr=6, thSAD=150, prefilter=3, RefineMotion=True, plane=0, chroma=False)
denoise = BM3D(deblock, ref=denoise, radius=2, sigma=1.3, CUDA=False)

denoise = lvf.rfs(denoise, deblock, OP)

# deband settings for the episode
deband = placebo_deband(denoise, 20, 2, 2, 0)

stronger_deband = placebo_deband(denoise, 24, [3.6, 3.5], 1, 0)

strongest_deband = placebo_deband(denoise, 24, [4, 2.5], 3, 0)

deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = lvf.rfs(deband, strongest_deband, (26762, 26844))

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
op_deband = lvf.rfs(op_stronger_deband, op_weak_deband, (opstart+1278, opstart+1346))

deband = lvf.rfs(deband, op_deband, OP)
deband = core.std.MaskedMerge(deband, denoise, everything_mask)

# graining
grain = Graigasm(
    thrs=[x << 8 for x in (45, 100)], 
    strengths=[(0.21, 0), (0.2, 0)], 
    sizes=[1.25, 1.25], sharps=[65, 60],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

regrained = core.noise.Add(grain, type=0, var=7, seed=69)
grain = lvf.rfs(grain, regrained, regrain_ranges)

final = depth(grain, 10)

from stgfunc import output as o

final.set_output()

o(gt, text=None)

o(rescale)

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/Mob\ Psycho\ 100\ III/Mob07.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-3 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.72 --rc-lookahead 120 --crf 12.5 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2 --ipratio 1.3 --pbratio 1.2 - -o /mnt/HDD1/mob/ep07/final.hevc
