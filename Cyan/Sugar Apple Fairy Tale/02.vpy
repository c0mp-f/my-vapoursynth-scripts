import vapoursynth as vs
import vardefunc as vdf
import vodesfunc as vof
import havsfunc as haf
import lvsfunc as lvf
import vstools as vst
import vsscale as vsc
import vsdenoise
import vsdehalo
import vsdeband
import vsmask
import vsaa

from vskernels import Spline64, Bicubic
from dfttest2 import DFTTest, Backend
from vstools import set_output as o
from vsdenoise import ccd, BM3DCPU
from vsdeband import Placebo

core = vs.core

OP = [(648, 2805)]
opstart = OP[0][0]

sraa_ranges = [
    (opstart+1273, opstart+1302), (3680, 3900), (4050, 4286)
]

taa_ranges = [
    (2868, 3209), (3680, 3900), (4510, 4580)
]

stronger_deband_ranges = [
    (10385, 10485), (16121, 16449), (23728, 23960), (29291, 29409)
]

weaker_deband_ranges = [
    (7977, 8840), (8257, 8495), (10385, 10485), (24983, 25014)
]

# sources
src = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Sugar Apple Fairy Tale - 02 (1080p) [89F39723].mkv")
src = vst.depth(src, 16)

srcy = vst.get_y(src)
srcy32 = vst.depth(srcy, 32)

preden = DFTTest(srcy, backend=Backend().CPU)

# AA
singlerater = vsaa.Eedi3SR(opencl=True)

aafunc = vsaa.Eedi3(beta=0.6, opencl=True)

aamask = vsmask.edge.Robinson5().edgemask(preden, multi=1.4).std.Binarize(10<<8).std.Inflate()

sraa = vsaa.upscaled_sraa(src, aafunc=singlerater)
sraa = core.std.MaskedMerge(src, sraa, aamask)

taa = vsaa.transpose_aa(src, aafunc)
taam = core.std.MaskedMerge(src, sraa, aamask)

faa = vsaa.fine_aa(src, taa=True, singlerater=singlerater)

aa = vsaa.clamp_aa(src, faa, taa, 1.5)
aa = vst.rfs(aa, sraa, sraa_ranges)
aa = vst.rfs(aa, taam, taa_ranges)

with vdf.YUVPlanes(aa) as caa:
    caa.U = vsaa.transpose_aa(caa.U, aafunc)
    caa.V = vsaa.transpose_aa(caa.V, aafunc)

aa = caa.clip

# denoise
cden = core.std.MaskedMerge(
    ccd(aa, thr=6, tr=1, matrix=1, mode=vsdenoise.CCDMode.BICUBIC_LUMA),
    ccd(aa, thr=3, tr=1, matrix=1, mode=vsdenoise.CCDMode.BICUBIC_LUMA),
    core.adg.Mask(aa.std.PlaneStats(), 48)
)
ref = vsdenoise.knl_means_cl(cden, strength=0.4, tr=2, sr=6, simr=6, channels=vsdenoise.ChannelMode.LUMA)
denoise = BM3DCPU(cden, ref=ref, sigma=[0.9, 0, 0], radius=2, matrix=1).clip

stronger_ref = vsdenoise.MVTools(cden, **vsdenoise.MVToolsPresets.SMDE).degrain(thSAD=(350, 150))
denoise_stronger = BM3DCPU(cden, ref=stronger_ref, sigma=2.3, radius=2, matrix=1).clip

denoise = vst.rfs(denoise, denoise_stronger, (23728, 23960))

dering = vsdehalo.edge_cleaner(denoise, strength=5, rmode=16)

# deband
detail_mask = vsdeband.deband_detail_mask(preden, brz=[925, 925])

deband = Placebo(radius=24, thr=[2.1, 1.2], grains=0, iterations=2).deband(dering)

stronger_deband = Placebo(radius=24, thr=3, grains=0, iterations=2).deband(dering)

weaker_deband = Placebo(radius=24, thr=[1.5, 1], grains=0, iterations=2).deband(dering)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = vst.rfs(deband, weaker_deband, weaker_deband_ranges)
deband = core.std.MaskedMerge(deband, dering, detail_mask)

grain = vdf.Graigasm(
    thrs=[x << 8 for x in (30, 85, 130)], 
    strengths=[(0.25, 0), (0.2, 0), (0.25, 0)], 
    sizes=[1.3, 1.2, 1.25], sharps=[70, 60, 75],
    grainers=[vdf.AddGrain(seed=1), vdf.AddGrain(seed=2, constant=True), vdf.AddGrain(seed=3, constant=True)]
).graining(deband)

final = vst.depth(grain, 10)

final.set_output()

o(src)
o(aa)
o(ref)
o(denoise)
o(dering)
o(deband)

o(detail_mask)

#o(linemask)

#o(core.adg.Mask(aa.std.PlaneStats(), 36))

#vspipe -c y4m ~/my-vapoursynth-scripts/Cyan/Sugar\ Apple\ Fairy\ Tale/02.vpy - | x265 --preset veryslow --bframes 12 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --no-rect --no-amp --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -1:-1 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.74 --rc-lookahead 60 --crf 13 --aq-mode 3 --aq-strength 0.8 --psy-rd 1.9 --psy-rdoq 2.2 --ipratio 1.4 --pbratio 1.3 --zones 23601,23686,q=8/23728,23960,q=6/24092,24108,q=8/24513,24582,q=8/24932,25275,b=1.2  - -o /mnt/HDD1/apple/ep02/final.hevc