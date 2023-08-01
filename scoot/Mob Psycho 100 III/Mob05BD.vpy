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
from vskernels import Catrom, ZewiaCubic, NoScale
from lvsfunc import stack_planes as sp
from vstools import set_output as o
from rekt import rektlvls

from filter import texture_mask, merge_chroma

core = vs.core

OP = [(3573, 5730)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [(31530, 33687)]

regrain_ranges = [
    (opstart+216, opstart+251), (opstart+755, opstart+764)
]

stronger_deband_ranges = [    
    (30175, 30317)
]

weaker_deband_ranges = [
    (1478, 1600), (10373, 10508), (16226, 16328), (18120, 18141)
] + ED

dpir_ranges = [
    (17552, 17597), (21929, 21998), (26251, 26272)
]

aa_ranges = [
    0
]

NO_EDGEFIX = OP
NO_DESCALE = OP

OP_STRONGER_DEBAND_RANGES = [
    (opstart+730, opstart+754), (opstart+937, opstart+1039), (opstart+1347, opstart+1483), 
    (opstart+1759, opstart+2031)
] + regrain_ranges

cr_merge = [
    (opstart+749, opstart+754), (17552, 17563)
]

aa_ranges = [
    (15802, 15897)
]

# muxtools setup
setup = mux.Setup("05")

jp = mux.src_file("/mnt/HDD1/Videos/Anime/TV/MOBPSYCHO100_3_DISC2/BDMV/STREAM/00001.m2ts", True)
src = vst.initialize_clip(jp.src)

cr = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 (01-12) (1080p) [Batch]/[SubsPlease] Mob Psycho 100 S3 - 05v2 (1080p) [CF520AEE].mkv", True)
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

linemask = core.std.Sobel(vst.get_y(preden)).std.Binarize(23<<8).std.Inflate().std.Inflate()
detail_mask = vsmasktools.detail_mask_neo(preden, detail_brz=0.01, lines_brz=0.02)
textures = texture_mask(preden, 2, points=[x * 256 for x in (0.75, 1.45, 4.5, 20)])

based_mask = vst.replace_ranges(core.std.Expr([detail_mask, textures], "x y max"), linemask, regrain_ranges + [OP_STRONGER_DEBAND_RANGES[0]] + OP_STRONGER_DEBAND_RANGES[2:])

# descale 
catrom = Catrom.descale(vst.get_y(rekt), 1280, 720)

rescale = vsc.Waifu2x.scale(catrom, 3840, 2160)

# rfactor=2
# based_aa is just constant configgings
aa = vsaa.Eedi3(alpha=0.125, beta=0.5, gamma=80, vthresh0=12, vthresh1=24, sclip_aa=None).aa(rescale)

stronger_aa = vsaa.upscaled_sraa(rescale, rfactor=1.1)
aa = vst.replace_ranges(aa, stronger_aa, aa_ranges)

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
    [[(opstart+490, opstart+556), (opstart+1347, opstart+1483)], 9],
    [dpir_ranges, 4]
]

deblock = vsd.dpir(aa, strength=0, zones=zones)
deblock = vst.replace_ranges(aa, deblock, dpir_ranges + OP)

cden = vsd.ccd(deblock, thr=4)
mv = vsd.MVTools(cden, **MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(120, 70))
denoise = vsd.BM3DCudaRTC(deblock, sigma=[1.4, 0], tr=2, profile=Profile.NORMAL).final(deblock, mv)

# deband
deband = vdb.Placebo(radius=28, thr=2.2, iterations=4).deband(denoise)

weaker_deband = vdb.Placebo(radius=28, thr=1.6, iterations=4).deband(denoise)

stronger_deband = vdb.Placebo(radius=28, thr=3.5, iterations=4).deband(denoise)

OP_STRONGER_DEBAND = vdb.Placebo(radius=24, thr=4, iterations=4).deband(denoise)

deband = vst.replace_ranges(deband, stronger_deband, stronger_deband_ranges)
deband = vst.replace_ranges(deband, weaker_deband, weaker_deband_ranges)
deband = vst.replace_ranges(deband, OP_STRONGER_DEBAND, OP_STRONGER_DEBAND_RANGES)

deband = core.std.MaskedMerge(deband, denoise, based_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(1.8, 1.7), size=1.3, dynamic=True, luma_scaling=12, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.9, 1.8), size=1.3, dynamic=False, luma_scaling=10, seed=2)

grain = vst.replace_ranges(
    grain, vdb.AddNoise.GAUSS.grain(deband, strength=(5, 1), seed=69),
    regrain_ranges
)

final = vst.finalize_clip(grain)
final = core.lsmas.LWLibavSource("/mnt/HDD1/mob/ep05/lossless.mkv") + core.lsmas.LWLibavSource("/mnt/HDD1/mob/ep05/lossless2.mkv")

if __name__ == "__main__" and final.num_frames == src.num_frames:
    settings = mux.settings_builder_x265(
        crf=13, 
        aq_strength=0.7, psy_rd=2.05, psy_rdoq=2.05, chroma_qpoffsets=-2,
        qcomp=0.72, rd=3,
        preset="veryslow", 
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    #mux.LosslessX264(mux.LosslessPreset.COMPRESSION).encode(final[23259:], "/mnt/HDD1/mob/ep05/lossless2.mkv")
    v = mux.x265(settings).encode(final, "/mnt/HDD1/mob/ep05/final")
else:
    final.set_output()
    #o(core.lsmas.LWLibavSource("/mnt/HDD1/mob/ep03/final.mkv"))

    o(src)

    o(cr)
    o(rescale)
    o(aa)
    o(denoise)
    o(deband)

    o(sp(src))
    o(sp(deband))

    o(linemask)
    o(based_mask)
