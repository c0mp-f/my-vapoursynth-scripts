import vapoursynth as vs
import vsdenoise as vsd
import vsdehalo as vsdh
import awsmfunc as awf
import lvsfunc as lvf
import vstools as vst
import vsmasktools

from vstools import set_output as o
from vsdeinterlace import sivtc
from vsdeband import AddGrain
from rekt import rektlvls

core = vs.core
sp = lvf.comparison.stack_planes

deblock_ranges = [
    (13120, 13539), (24614, 24804), (25054, 25530), (30595, 30612),
    (31945, 32146), (35740, 35954), (36027, 36266), (45226, 46190),
    (46609, 46716), (47913, 47973), (48383, 48669), (60555, 60690), 
    (80335, 80398)
]

def nnedi(clip: vs.VideoNode) -> vs.VideoNode:
    up = vst.get_y(clip).znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3)
    up = up.std.Transpose()
    up = up.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3)
    return up.std.Transpose()

# source import
srcs = [core.d2v.Source(f"fumo_conspiracy{i}.d2v") for i in range(1, 6)]

src = srcs[0] + srcs[1] + srcs[2] + srcs[3] + srcs[4]
src = vst.initialize_clip(src)

ivtc = sivtc(src, pattern=1, tff=False)

# IVTC trolling
ivtc = ivtc[:21638] + ivtc[21639] + ivtc[21639:]
ivtc = ivtc[:45226] + ivtc[45227] + ivtc[45227:]

# edgefixing
rekt = rektlvls(
    ivtc, 
    rownum=[477, 478], rowval=[3, 2],
)
edgefix = awf.bbmod(rekt, bottom=1)

# denoise
denoise = vsd.nl_means(edgefix, strength=0.6, tr=2, sr=4, planes=[1, 2])

# dehaloing
fd_args = dict(rx=4, thmi=50, thma=100, thlimi=70, thlima=128, edgemask=vsmasktools.edge.Robinson5())

up = nnedi(denoise)

blur = vsd.Prefilter.GAUSSBLUR2

dehalo = vsdh.dehalo_sigma(up, brightstr=1, darkstr=0.01, blur_func=blur(sigma=3.5))
dehalo = vsdh.dehalo_sigma(dehalo, brightstr=1, darkstr=0.2, lowsens=85, highsens=25).resize.Point(720, 480)

hmask = vsdh.fine_dehalo(up, **fd_args, show_mask=True).resize.Bicubic(720, 480)

dehalo = core.std.MaskedMerge(vst.get_y(denoise), dehalo, hmask)

dehalo = core.std.ShufflePlanes([dehalo, denoise], [0, 1, 2], vs.YUV)

# deringing
deringm = vsmasktools.edge.Prewitt().edgemask(vst.get_y(dehalo))
deringm = deringm.std.Binarize(50<<8).std.Maximum().std.Maximum().std.Inflate()

dpir_mask = vsd.dpir_mask(dehalo, low=7, high=14).resize.Point(format=vs.YUV420P16)

zones = [
    [[(30595, 30612), (48383, 48669)], 35],
    [[(13120, 13539), (60555, 60690)], 25],
    [[(24614, 24804), (25054, 25530), (80335, 80398)], 15]
]
dpir = vsd.dpir(vst.get_y(dehalo), strength=10, zones=zones)
dpir = core.std.ShufflePlanes([dpir, dehalo], [0, 1, 2], vs.YUV)

dpirRGB = vsd.dpir(dehalo, strength=10, zones=zones)

dpir = vst.replace_ranges(dpir, dpirRGB, [(30595, 30612), (35740, 35954), (48383, 48669), (80335, 80398)])

dpir = core.std.MaskedMerge(dpir, dehalo, dpir_mask)

dering = core.std.MaskedMerge(dehalo, dpir, deringm)

# attempt at dempegging high-motion scenes, and regraining
deblock = vsd.MVTools(dpir, **vsd.MVToolsPresets.NOISY).denoise(dpir, thSAD=150)
deblock = vsd.BM3DCudaRTC(deblock, sigma=1.5, tr=2).final(dpir, deblock)
regrain = AddGrain(strength=(1.15, 0), size=1.15, seed=1).grain(deblock)

deblock = vst.replace_ranges(dering, regrain, deblock_ranges) 

dering = vsdh.edge_cleaner(deblock, strength=3, hot=True)

final = vst.finalize_clip(dering)

final = core.lsmas.LWLibavSource("/mnt/HDD1/lupin/movie.mkv")

final.set_output()

o(ivtc)
o(denoise)
o(dehalo)
o(dering)
o(deringm)
o(sp(ivtc))
o(sp(final))

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/The\ Fumo\ Conspiracy.vpy - | \
#x265 - \
#--y4m --preset veryslow --no-strong-intra-smoothing --no-sao --no-open-gop --no-cutree --no-amp --no-rect --high-tier --aud --repeat-headers \
#--input-depth 10 --output-depth 10 --range limited --colormatrix 6 --colorprim 6 --transfer 6 --sar 80:79 \
#--bframes 16 --rd 3 --merange 57 --deblock -2:-2 --rc-lookahead 60 \
#--aq-mode 3 --aq-strength 0.7 \
#--crf 11 --qcomp 0.74 --cbqpoffs -3 --crqpoffs -3 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 \
#-o /mnt/HDD1/lupin/movie.hevc
