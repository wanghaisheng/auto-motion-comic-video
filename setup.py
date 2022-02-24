from setuptools import setup, find_packages
import re


with open("README.md", "r",encoding="utf-8") as f:
    long_description = f.read()


with open("auto_motion_comic_video/__init__.py") as f:
    version = re.search(
        r"""^__version__\s*=\s*['"]([^\'"]*)['"]""", f.read(), re.MULTILINE
    ).group(1)


setup(
    name="auto_motion_comic_video",
    version=version,
    author="wanghaisheng",
    author_email="edwin.uestc@gmail.com",
    description="动态漫画火遍国内，以ace attorney bot为基础探索自动化动态漫画视频创建.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/wanghaisheng/auto-motion-comic-video",
    download_url="https://github.com/wanghaisheng/auto-motion-comic-video/tarball/v" + version,
    packages=["auto_motion_comic_video"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["selenium"],
    include_package_data=True,
    python_requires=">=3.6",
)
