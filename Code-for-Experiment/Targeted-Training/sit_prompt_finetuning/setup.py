from setuptools import setup, find_packages
from ama_prof_divi_hifigan import __VERSION__, __AUTHOR__

def get_long_description():
    with open("Code-for-MuChin-AP(AnnotationPlatform)/backend_api/Code-for-MuChin-AP(AnnotationPlatform)/backend_api/README.md", "r", encoding="utf-8") as fh:
        return fh.read()

def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup(
    name="music-dit2",
    version=__VERSION__,
    author=__AUTHOR__,
    url="https://github.com/ama-prof-divi-ai/music-dit2",
    description="DiT model for music Generation.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformers",
        "diffusion",
        "dit",
        "vae",
        "hifigan",
        "mel",
        "music"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
        "License :: OSI Approved :: MIT License 2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ]
)