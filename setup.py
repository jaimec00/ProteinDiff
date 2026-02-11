from setuptools import setup

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    wf = "proteus/model/tokenizer/wave_func_tokenizer"

    ext_modules = [
        CUDAExtension(
            name="proteus.model.tokenizer.wave_func_tokenizer.learn_aa._C",
            sources=[
                f"{wf}/learn_aa/wf_embedding_learn_aa_if.cpp",
                f"{wf}/learn_aa/wf_embedding_learn_aa_kernel.cu",
            ],
        ),
        CUDAExtension(
            name="proteus.model.tokenizer.wave_func_tokenizer.static_aa._C",
            sources=[
                f"{wf}/static_aa/wf_embedding_static_aa_if.cpp",
                f"{wf}/static_aa/wf_embedding_static_aa_kernel.cu",
            ],
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    pass

setup(ext_modules=ext_modules, cmdclass=cmdclass)
