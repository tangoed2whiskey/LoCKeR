import multiprocessing
import os
import setuptools

VERSION = "0.1.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

kwargs = {
    "name": "LoCKeR",
    "version": VERSION,
    "description": ("Local Covariance Kernel Regression"),
    "url": "https://github.com/tangoed2whiskey/LoCKeR",
    "author": "Tom Whitehead",
    "author_email": "tom@intellegens.ai",
    "classifiers": [
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    "install_requires": [
        "scikit-learn",
    ],
    "license": "GPL 3",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "package_data": {"LoCKeR": ["py.typed"]},
    "packages": setuptools.find_namespace_packages(include=["locker*"]),
    "python_requires": ">=3.6",
    "setup_requires": ["wheel"],
    "zip_safe": False,
}

if os.getenv("TARGET", "unknown") == "cython":
    # Add Cython kw args
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

    # empty list to prevent .py files ending up in wheel
    kwargs["packages"] = []

    kwargs["cmdclass"] = {
        "build_ext": build_ext,
    }
    kwargs["ext_modules"] = cythonize(
        [
            setuptools.extension.Extension("locker.*", ["locker/*.py"]),
            setuptools.extension.Extension(
                "locker.analysis.*", ["locker/analysis/*.py"]
            ),
            setuptools.extension.Extension(
                "locker.ml_methods.*", ["locker/ml_methods/*.py"]
            ),
            setuptools.extension.Extension("locker.pca.*", ["locker/pca/*.py"]),
        ],
        build_dir="build",
        compiler_directives={
            "always_allow_keywords": True,
            "language_level": "3",
            "emit_code_comments": False,
        },
        # parallelise .py->.c compilation
        nthreads=multiprocessing.cpu_count(),
    )

setuptools.setup(**kwargs)
