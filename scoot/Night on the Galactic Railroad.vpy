import vapoursynth as vs
import vsdehalo as vdh
import awsmfunc as awf 
import vstools as vst

from vsdenoise import MVTools, MVToolsPresets, BM3DCudaRTC, nl_means, ccd, Prefilter, Profile
from vskernels import Catrom, Point, Lanczos
from vstools import set_output as o

from vsmuxtools import Setup, src_file, settings_builder_x265, mux
from vsmuxtools import x265, do_audio, Opus, SubFile, Chapters
from vsmuxtools import TmdbConfig

core = vs.core

### SOURCES
setup = Setup("it's a movie")

US = src_file("/mnt/HDD1/Videos/Anime/Movies/Night on The Galactic Railroad/BDMV/STREAM/00000.m2ts", True)
src = vst.initialize_clip(US.src)
crop = core.std.Crop(src, left=16, right=16)

preden = nl_means(crop, strength=8, tr=1, planes=0)

### EDGEFIXING
bb = awf.bbmod(crop, cBottom=3, cTop=2)

### DENOISE
cden = ccd(bb, thr=5.5)

deblock = core.deblock.Deblock(cden, quant=16.5, aoffset=2, boffset=3)
mvtools = MVTools(deblock, tr=3, **MVToolsPresets.NOISY).denoise(deblock, thSAD=(200, 65))
denoise = BM3DCudaRTC(cden, sigma=[1.3, 0], tr=2).final(cden, mvtools)

denoise = vst.replace_ranges(denoise, deblock, [(61633, 62112)])

### DEHALO
fd_args = dict(rx=2, thlimi=60, thlima=100, thmi=30, thma=100)

dehalo = vdh.dehalo_sigma(denoise, sigma=2.7, pre_ss=2)

halo_mask = vdh.fine_dehalo.mask(preden, **fd_args)

dehalo = core.std.MaskedMerge(denoise, dehalo, halo_mask.resize.Point(format=vs.YUV420P16))

final = vst.finalize_clip(dehalo)

if __name__ == "__main__":
    settings = settings_builder_x265(
        crf=18.4, 
        aq_strength=0.74, psy_rd=2.1, psy_rdoq=2.4, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow",
        deblock=[-1, 0], bframes=16,
        rect=False
    )
    v = x265(settings).encode(final)
    
    us_audio = do_audio(US, track=0).to_track("English 2.0 Opus", "en")
    
    jp_audio = do_audio(US, track=1, encoder=Opus(bitrate=384)).to_track("Japanese 6.0 Opus", "jp")
    
    commentary = do_audio(US, track=3).to_track("English Commentary 2.0 AC3", "en")
    
    dialogue = SubFile("/mnt/HDD1/notgr/notgr.ass").clean_garbage().change_layers()

    signs = SubFile("/mnt/HDD1/notgr/ts.ass")

    subs = SubFile([dialogue.file, signs.file])
    fonts = subs.collect_fonts()

    subs = subs.to_track("[scoot] Full Dialogue", "en", default=True, forced=False)
    signs = signs.to_track("[scoot] Signs", "en", default=False, forced=True)

    chapters = Chapters("/mnt/HDD1/notgr/chapters.xml")

    mux(
        v, jp_audio, us_audio, commentary, 
        subs, signs, chapters, *fonts, tmdb=TmdbConfig(37585, movie=True)
    )


else:

    final.set_output()

    o(enc)

    o(bb)
    o(cden)
    o(mvtools)
    o(denoise)
    o(dehalo)

    o(halo_mask)
