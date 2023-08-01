import vapoursynth as vs
import vsmuxtools as mux
import vsdenoise as vsd
import vsdeband as vdb
import awsmfunc as awf
import vstools as vst 
import vsscale as vsc
import vsmasktools
import vsaa

from vsdenoise import MVToolsPresets, Prefilter, Profile
from lvsfunc import stack_planes as sp
from vstools import set_output as o
from vskernels import Catrom, Mitchell, ZewiaCubic
from rekt import rektlvls

from filter import texture_mask, merge_chroma

core = vs.core

OP = [(768, 2924)]
opstart = OP[0][0]

regrain_ranges = [
    (opstart+216, opstart+251), (opstart+755, opstart+764)
]

stronger_deband_ranges = [
    (25741, 25904), (26170, 26201), (32109, 32243), (32680, 32808)
]

NO_EDGEFIX = OP
NO_DESCALE = OP

OP_STRONGER_DEBAND_RANGES = [
    (opstart+730, opstart+754), (opstart+937, opstart+1039), (opstart+1347, opstart+1483), (opstart+1759, opstart+2031)
] + regrain_ranges

cr_merge = [
    (opstart+749, opstart+754)
]

# muxtools setup
setup = mux.Setup("01")

src = mux.src_file("/mnt/HDD1/Videos/Anime/TV/MOBPSYCHO100_3_DISC1/BDMV/STREAM/00001.m2ts", True)
src = vst.initialize_clip(src.src)

cr = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 (01-12) (1080p) [Batch]/[SubsPlease] Mob Psycho 100 S3 - 01v2 (1080p) [95315F0C].mkv", True)
cr = vst.initialize_clip(cr.src)

src = vst.replace_ranges(src, cr, cr_merge)
# edgefixing
rekt = rektlvls(
    src,
    colnum=[0, 1919], colval=[15, 15],
    rownum=[0, 1079], rowval=[17, 16]
)
bb = awf.bbmod(rekt, top=1)
rekt = vst.replace_ranges(bb, src, NO_EDGEFIX)

preden = vsd.DFTTest(rekt, sigma=5).denoise(rekt)

linemask = core.std.Sobel(vst.get_y(preden)).std.Binarize(30<<8).std.Inflate().std.Inflate()
detail_mask = vsmasktools.detail_mask_neo(preden, detail_brz=0.01, lines_brz=0.02)
textures = texture_mask(preden, 2, points=[x * 256 for x in (0.75, 1.65, 4.5, 20)])

based_mask = vst.replace_ranges(core.std.Expr([detail_mask, textures], "x y max"), linemask, regrain_ranges + [OP_STRONGER_DEBAND_RANGES[0]] + OP_STRONGER_DEBAND_RANGES[2:])

# descale 
catrom = Catrom.descale(vst.get_y(rekt), 1280, 720)

rescale = vsc.Waifu2x.scale(catrom, 3840, 2160)

# rfactor=2
# based_aa is just constant configgings
aa = vsaa.Eedi3(alpha=0.14, beta=0.55, gamma=55, vthresh0=12, vthresh1=24, sclip_aa=None).aa(rescale)

aa = Catrom.scale(aa, 1920, 1080)
aa = merge_chroma(aa, rekt)

# purely for the sake of comparison
rescale = Catrom.scale(rescale, 1920, 1080)
rescale = merge_chroma(rescale, rekt)
rescale = core.std.MaskedMerge(rekt, rescale, linemask)

aa = core.std.MaskedMerge(rekt, aa, linemask)

error = vsc.descale_detail_mask(rekt, Catrom().scale(catrom, 1920, 1080), thr=0.03)

aa = core.std.MaskedMerge(aa, rekt, error)

aa = vst.replace_ranges(aa, rekt, NO_DESCALE)

# denoise
zones = [
    [[(opstart+749, opstart+764)], 30],
    [[(opstart+216, opstart+251)], 20],    
    [[(opstart+490, opstart+556), (opstart+1347, opstart+1483)], 9]
]

deblock = vsd.dpir(aa, strength=0, zones=zones)
deblock = vst.replace_ranges(aa, deblock, OP)

mv = vsd.MVTools(deblock, **MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)).denoise(deblock, thSAD=(100, 70))
denoise = vsd.BM3DCudaRTC(deblock, sigma=[1.3, 0], tr=2, profile=Profile.NORMAL).final(deblock, mv)

# deband
deband = vdb.Placebo(radius=28, thr=2, iterations=4).deband(denoise)

stronger_deband = vdb.Placebo(radius=28, thr=3, iterations=4).deband(denoise)

OP_STRONGER_DEBAND = vdb.Placebo(radius=24, thr=4, iterations=4).deband(denoise)

deband = vst.replace_ranges(deband, stronger_deband, stronger_deband_ranges)
deband = vst.replace_ranges(deband, OP_STRONGER_DEBAND, OP_STRONGER_DEBAND_RANGES)

deband = core.std.MaskedMerge(deband, denoise, based_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(2, 1.8), size=1.2, dynamic=True, luma_scaling=14, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.9, 1.9), size=1.25, dynamic=False, luma_scaling=10, seed=2)

grain = vst.replace_ranges(
    grain, vdb.AddNoise.GAUSS.grain(deband, strength=(5, 1), seed=69),
    regrain_ranges
)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13, 
        aq_strength=0.7, psy_rd=2.05, psy_rdoq=2.15, chroma_qpoffsets=-2,
        qcomp=0.73, rd=3,
        preset="veryslow", 
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    #mux.FFV1(mux.LosslessPreset.COMPRESSION).encode(final, "/mnt/HDD1/mob/ep01/lossless.mkv")
    mux.x265(settings).encode(final, "/mnt/HDD1/mob/ep01/final.265")
else: 
    final.set_output()

    o(src)
    o(rescale)
    o(aa)
    o(denoise)
    o(deband)

    o(sp(src))
    o(sp(denoise))

    o(linemask)
    o(based_mask)
