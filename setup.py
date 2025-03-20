import setuptools

setuptools.setup(
    name="extension_cosyvoice",
    packages=setuptools.find_namespace_packages(),
    version="0.1.3",
    author="rsxdalv",
    description="CosyVoice: A TTS solution for fluent and natural speech synthesis.",
    url="https://github.com/rsxdalv/extension_cosyvoice",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        "cosyvoice @ git+https://github.com/rsxdalv/CosyVoice@main",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
