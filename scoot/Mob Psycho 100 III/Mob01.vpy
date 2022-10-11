import vapoursynth as vs 
import vardefunc as vdf
import havsfunc as haf
import awsmfunc as awf
import lvsfunc as lvf

from vsscale import descale_detail_mask, ssim_downsample
from lvsfunc.comparison import stack_planes as sp
from debandshit import placebo_deband, dumb3kdb
from vardefunc import Graigasm, AddGrain
from vsutil import depth, insert_clip
from EoEfunc.denoise import BM3D
from filter import texture_mask
from vskernels import Bicubic 
from rekt import rektlvls
from vsdpir import DPIR

core = vs.core

def OP_filtering(clip: vs.VideoNode) -> vs.VideoNode:
    
    rgbs = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=1)
    denoise = core.wnnm.WNNM(rgbs)
    deblock = DPIR(denoise, strength=20, task="deblock")
    deblock = core.resize.Bicubic(deblock, format=vs.YUV420P16, matrix=1)

    crop = deblock.std.Crop(0, 0, 138, 138)
    bb = awf.bbmod(crop, top=1, bottom=1, cBottom=1, cTop=1)
    letterbox = bb.std.AddBorders(top=138, bottom=138)
    
    op = lvf.rfs(deblock, letterbox, (354, 594))

    return op

OP = [(768, 2924)]
opstart = OP[0][0]
regrain_ranges = [
    (opstart+216, opstart+248), (opstart+755, opstart+764), (opstart+850, opstart+866)
]

NO_DESCALE = OP
NO_EDGEFIX = OP

# setup
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[NC-Raws] Mob Psycho 100 III - 01 (CR 1920x1080 AVC AAC MKV) [528D19AC].mkv")
src = depth(gt, 16)

preden = core.dfttest.DFTTest(src, sigma=15)

detail_mask = lvf.mask.detail_mask(preden, rad=2, brz_a=0.015, brz_b=0.015)
textures = texture_mask(preden, radius=2, points=[x * 256 for x in (0.8, 1.25, 5, 12)])
textures = lvf.rfs(textures, core.std.BlankClip(textures), OP) 

everything_mask = core.std.Expr([detail_mask, textures], "x y max")

# sceneflt. 
op = OP_filtering(src[768:2924])
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
error_mask = descale_detail_mask(rekt, catrom_rescale, threshold=10*257)

rescale = core.std.MaskedMerge(rescale, rekt, emask)

# denoising
rgbs = Bicubic().resample(rescale, vs.RGBS, matrix_in=1)
deblock = DPIR(rgbs, strength=10, task="deblock")
deblock = Bicubic().resample(deblock, vs.YUV420P16, matrix=1)
deblock = lvf.rfs(rescale, deblock, (25535, 26169))

denoise = haf.SMDegrain(deblock, tr=6, thSAD=100, RefineMotion=True, prefilter=3)
denoise = BM3D(deblock, ref=denoise, sigma=1, CUDA=False)

# deband
averaged_deband = core.average.Mean([
    denoise,
    placebo_deband(denoise, 4, 1.8, 2, 0),
    placebo_deband(denoise, 6, 2.4, 2, 0),
    placebo_deband(denoise, 10, 2, 2, 0),
    placebo_deband(denoise, 12, 4, 1, 0),
]) 

deband = placebo_deband(denoise, 16, [2.2, 2.3], 1, 0)
deband_harder = placebo_deband(denoise, 16, 2.5, 2, 0)
deband = lvf.rfs(deband, deband_harder, (25741, 25904))

op_weak_deband = core.neo_f3kdb.Deband(averaged_deband, y2=2, cb2=2, cr2=2, range=30, grainy=0, grainc=0)
op_stronger_deband = placebo_deband(averaged_deband, 26, [3, 2.5], 1, 0)
op_deband = lvf.rfs(op_stronger_deband, op_weak_deband, (opstart+1278, opstart+1346))

deband = lvf.rfs(deband, op_deband, OP)
deband = core.std.MaskedMerge(deband, denoise, everything_mask)

# graining
grain = Graigasm(
    thrs=[x << 8 for x in (40, 140)], 
    strengths=[(0.27, 0), (0.2, 0)], 
    sizes=[1.2, 1.2], sharps=[60, 60],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

regrained = core.noise.Add(grain, type=0, var=7, seed=69)
grain = lvf.rfs(grain, regrained, regrain_ranges)

final = depth(grain, 10)

from stgfunc import output as o 

final.set_output()

o(gt)
o(rescale,text=None)
o(denoise,text=None)
o(deband)

o(everything_mask)

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/Mob/Mob01.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.74 --rc-lookahead 120 --crf 13 --no-sao --aq-mode 3 --aq-strength 0.78 --psy-rd 2.1 --psy-rdoq 2.1 --ipratio 1.3 --pbratio 1.2 - -o /mnt/HDD1/mob/ep01/final.hevc
