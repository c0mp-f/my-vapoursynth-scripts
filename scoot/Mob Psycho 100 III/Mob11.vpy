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

OP = [(1942, 4100)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [(31531, 33688)]

regrain_ranges = [
    (opstart+216, opstart+248), (opstart+755, opstart+764), (opstart+850, opstart+866)
]
dpir_ranges = [
    (5599, 5646), (5754, 5815), (7144, 7204), (8583, 8749),
    (8947, 9111), (10645, 10949), (11399, 11431), (12804, 12874),
    (13344, 13522), (24373, 24443), (24510, 24866), (25934, 25951)
] + ED

weaker_deband_ranges = [(11165, 11223)]

stronger_deband_ranges = [
    (6060, 6087), (7017, 7143), (10811, 10909), (10928, 10949),
    (10988, 11058), (11432, 11483), (12804, 12874),(13945, 14172), 
    (15994, 16118), (24435, 24443), (25102, 25158), (25197, 25282)
]

strongest_deband_ranges = [
    (13344, 13522), (20012, 20124), (24510, 24866), (25934, 25951), (30435, 30458)
]

aa_ranges = [
    (4385, 4488)
]

NO_DESCALE = OP + ED
NO_EDGEFIX = OP

# setup
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 - 11 (1080p) [EDE5CCEB].mkv")
bg = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[NC-Raws] 路人超能100 III - 11 (B-Global 3840x2160 HEVC AAC MKV) [528468FD].mkv")

src = depth(gt, 16)

bg = depth(bg, 16)
bg = Bicubic().scale(bg, 1920, 1080)

preden = core.dfttest.DFTTest(src, sigma=5)

detail_mask = lvf.mask.detail_mask(preden, brz_a=0.015, brz_b=0.015, rad=3,)
textures = texture_mask(preden, 2, points=[x * 256 for x in (1, 1.5, 5, 15)])
textures = vst.rfs(textures, core.std.BlankClip(textures), [(opstart+1347, opstart+1483), (opstart+1759, opstart+2031), (30435, 30458)]) 

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
merge = vst.rfs(rescale, merge, [(5754, 5815), (12804, 12874), (20012, 20124)])

# denoising
rgbs = Bicubic().resample(merge, vs.RGBS, matrix_in=1)
deblock = DPIR(rgbs, strength=5, task="deblock")
deblock = Bicubic().resample(deblock, vs.YUV420P16, matrix=1)

deblock = vst.rfs(merge, deblock, dpir_ranges)

cden = ccd(deblock, 6)
denoise = haf.SMDegrain(cden, tr=6, thSAD=150, prefilter=3, RefineMotion=True, plane=0, chroma=False)
denoise = BM3D(deblock, ref=denoise, radius=2, sigma=1.2, CUDA=False)

denoise = vst.rfs(denoise, deblock, OP)

# AA
eedi = Eedi3SR()
eedi.alpha = 0.5
eedi.beta = 0.25
eedi.gamma = 40

aa = upscaled_sraa(denoise, aafunc=eedi)

sm = squaremask(aa, 230, 346, 1120, 734)

aa = core.std.MaskedMerge(denoise, aa, sm)
aa = vst.rfs(denoise, aa, aa_ranges)

# deband settings for the episode
deband = placebo_deband(aa, 20, 1.5, 1, 0)

weaker_deband = placebo_deband(aa, 20, 1, 1, 0)

stronger_deband = placebo_deband(aa, 28, 2, 2, 0)

strongest_deband = placebo_deband(aa, 28, 3, 2, 0)

deband = vst.rfs(deband, weaker_deband, weaker_deband_ranges)
deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = vst.rfs(deband, strongest_deband, strongest_deband_ranges)

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
deband = core.std.MaskedMerge(deband, aa, everything_mask)

# graining
grain = Graigasm(
    thrs=[x << 8 for x in (50, 75, 130)], 
    strengths=[(0.25, 0), (0.2, 0), (0.25, 0)], 
    sizes=[1.3, 1.2, 1.3], sharps=[65, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

regrained = core.noise.Add(grain, type=0, var=7, seed=69)
grain = vst.rfs(grain, regrained, regrain_ranges)

final = depth(grain, 10)

from stgfunc import output as o

final.set_output()

o(src)
o(rescale)
o(aa)
o(deband)

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/Mob\ Psycho\ 100\ III/Mob11.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-3 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.73 --rc-lookahead 120 --crf 14 --no-sao --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2 --ipratio 1.3 --pbratio 1.2 - -o /mnt/HDD1/mob/ep11/final.hevc