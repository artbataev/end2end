import os
import platform
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        try:
            import torch
        except ImportError:
            sys.stderr.write("Pytorch is required to build this package\n")
            sys.exit(-1)

        self.pytorch_dir = os.path.dirname(torch.__file__)

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
                      "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
                      # "-DPYTHON_EXECUTABLE=".format(sys.executable),
                      ]

        config = "Debug" if self.debug else "Release"
        build_args = ["--config", config]

        if platform.system() == "Darwin":
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + config]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(extdir))
        self.spawn(["cmake", ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."])
        os.chdir(cwd)


setup(
    name="pytorch_end2end",
    version="0.0.1",
    description="Library with end-to-end losses and decoders",
    author="Vladimir Bataev",
    author_email="artbataev@gmail.com",
    url="https://artbataev.github.io/end2end/",
    packages=find_packages(),
    license="MIT",
    install_requires=["numpy", "numba", "tqdm"],
    ext_modules=[CMakeExtension("pytorch_end2end")],
    cmdclass={"build_ext": CMakeBuild, }
)
