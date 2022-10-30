import vapoursynth as vs 
import vardefunc as vdf
import havsfunc as haf
import awsmfunc as awf
import lvsfunc as lvf
import vsmask

from vsaa import Eedi3, Eedi3SR, upscaled_sraa, clamp_aa, transpose_aa, Eedi3SS
from vsutil import depth, insert_clip, get_y, split, join
from vsscale import descale_detail_mask, ssim_downsample
from lvsfunc.comparison import stack_planes as sp
from debandshit import placebo_deband, dumb3kdb
from vardefunc import Graigasm, AddGrain
from vsaa import transpose_aa, Eedi3
from EoEfunc.denoise import BM3D
from filter import texture_mask
from vskernels import Bicubic
from rekt import rektlvls
from vsdpir import DPIR
from ccd import ccd

core = vs.core

def OP_filtering(clip: vs.VideoNode) -> vs.VideoNode:
    
    denoise = haf.SMDegrain(clip, tr=6, thSAD=260, RefineMotion=True)
    denoise = depth(denoise, 32)
    denoise = core.wnnm.WNNM(denoise, sigma=[1, 1.25, 1.25])
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

OP = [(1296, 3453)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [(0, 1)]

regrain_ranges = [
    (opstart+216, opstart+248), (opstart+755, opstart+764), (opstart+850, opstart+866)
]
dpir_ranges = [
    (11234, 11479), (12047, 12130), (12794, 12877), (25490, 25602),
    (28492, 28659), (28738, 29020), (33798, 33983)
] + ED
stronger_deband_ranges = [
    (11234, 11479), (17329, 17388), (17870, 17907), (19074, 19193), 
    (27672, 27851), (28777, 28812), (33920, 33983)
]
aa_ranges = [
    (52, 77), (15802, 15897)
]

NO_DESCALE = OP + ED
NO_EDGEFIX = OP

# setup
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 - 04 (1080p) [495161CB].mkv")
src = depth(gt, 16)

preden = core.dfttest.DFTTest(src, sigma=15)

detail_mask = lvf.mask.detail_mask(preden, rad=2, brz_a=0.015, brz_b=0.015)
textures = texture_mask(preden, radius=3, points=[x * 256 for x in (0.7, 1.2, 5, 15)])
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

# scenefiltered AA
sraafunc = Eedi3SR()
sraafunc.gamma = 80

aa = upscaled_sraa(rescale, aafunc=sraafunc)

lm = vsmask.edge.ExSobel().edgemask(get_y(preden))
lm = lm.std.Binarize(110<<8).std.Inflate()

aa = core.std.MaskedMerge(rescale, aa, lm)
aa = lvf.rfs(rescale, aa, aa_ranges)

# denoising
rgbs = Bicubic().resample(aa, vs.RGBS, matrix_in=1)
deblock = DPIR(rgbs, strength=8, task="deblock")
deblock = Bicubic().resample(deblock, vs.YUV420P16, matrix=1)

deblock = lvf.rfs(aa, deblock, dpir_ranges)

cden = ccd(deblock, 4)
cden = lvf.rfs(cden, ccd(deblock, 8), (28777, 28812))
denoise = haf.SMDegrain(cden, tr=6, thSAD=150, prefilter=3, RefineMotion=True)
denoise = BM3D(deblock, ref=denoise, radius=2, sigma=1.5, CUDA=False)

denoise = lvf.rfs(denoise, deblock, OP)

# deband settings for the episode
deband = placebo_deband(denoise, 26, 1.2, 1, 0)

stronger_deband = placebo_deband(denoise, 24, [2, 2.5], 1, 0)

deband = lvf.rfs(deband, stronger_deband, stronger_deband_ranges)

# deband settings for the OP
averaged_deband = core.average.Mean([
    denoise,
    placebo_deband(denoise, 4, 1.8, 2, 0),
    placebo_deband(denoise, 6, 2.4, 2, 0),
	placebo_deband(denoise, 10, 2, 2, 0),
    placebo_deband(denoise, 12, 2, 1, 0),
]) 

op_weak_deband = core.neo_f3kdb.Deband(averaged_deband, y2=2, cb2=2, cr2=2, range=30, grainy=0, grainc=0)
op_stronger_deband = placebo_deband(averaged_deband, 26, [1.8, 2], 1, 0)
op_deband = lvf.rfs(op_stronger_deband, op_weak_deband, (opstart+1278, opstart+1346))

deband = lvf.rfs(deband, op_deband, OP)
deband = core.std.MaskedMerge(deband, denoise, everything_mask)

# graining
grain = Graigasm(
    thrs=[x << 8 for x in (40, 120)], 
    strengths=[(0.2, 0), (0.2, 0)], 
    sizes=[1.25, 1.25], sharps=[60, 60],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

regrained = core.noise.Add(grain, type=0, var=7, seed=69)
grain = lvf.rfs(grain, regrained, regrain_ranges)

final = depth(grain, 10)

from stgfunc import output as o 

final.set_output()

o(gt)
o(deband)
o(op)

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/Mob\ Psycho\ 100\ III/Mob04.vpy - | x265 --preset veryslow --bframes 10 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.72 --rc-lookahead 120 --crf 11 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2 --ipratio 1.3 --pbratio 1.2 - -o /mnt/HDD1/mob/ep04/final.hevc
