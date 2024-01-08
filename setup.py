import os
import sys
import tempfile
import setuptools

from importlib import import_module
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler


__version__ = '0.5.0'
dependencies = ['numpy>=1.10.0', 'pybind11>=2.2.3']


class PostponedIncludeGetter:
    def __init__(self, module_name):
        self.module_name = module_name

    def __str__(self):
        mod = import_module(self.module_name)
        return mod.get_include()


# compatibility when run in python_bindings
bindings_dir = 'python_bindings'
if bindings_dir in os.path.basename(os.getcwd()):
    source_files = ['./bindings.cpp']
    include_dir = '../hnswlib/'
else:
    source_files = ['./python_bindings/bindings.cpp']
    include_dir = './hnswlib/'


ext_modules = [
    Extension(
        'hnswlib',
        source_files,
        include_dirs=[
            include_dir,
            PostponedIncludeGetter('numpy'),
            PostponedIncludeGetter('pybind11'),
        ],
        libraries=[],
        language='c++',
        extra_objects=[],
    ),
]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3', '-march=native', '-Wno-reorder'],
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append("-fopenmp")
        link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass

        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(self.cpp_flag())
            if self.compiler_has_flag('-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))

        super().build_extensions()

    # As of Python 3.6, CCompiler has a `has_flag` method.
    # cf http://bugs.python.org/issue26689
    def compiler_has_flag(self, flag):
        """Return a boolean indicating whether a flag name is supported on
        the specified compiler.
        """
        with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
            f.write('int main (int argc, char **argv) { return 0; }')
            try:
                objects = self.compiler.compile([f.name], extra_postargs=[flag], output_dir=os.path.dirname(f.name))
            except setuptools.distutils.errors.CompileError:
                return False
            for obj in objects:
                os.unlink(obj)
        return True

    def cpp_flag(self):
        """Return the -std=c++[11/14] compiler flag.
        The c++14 is preferred over c++11 (when it is available).
        """
        for cpp_version in (14, 11):
            cpp_flag = f'-std=c++{cpp_version}'
            if self.compiler_has_flag(cpp_flag):
                return cpp_flag
        raise RuntimeError('Unsupported compiler: at least C++11 support is needed!')


setup(
    name='hnswlib',
    version=__version__,
    description='hnswlib',
    author='Yury Malkov and others',
    url='https://github.com/yurymalkov/hnsw',
    long_description="""hnsw""",
    ext_modules=ext_modules,
    install_requires=dependencies,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
