import setuptools

setuptools.setup(
    name="ITCL-QUANTIZATION-TOOLKIT-INFERENCE-ENGINE",
    version="0.0.5.2",
    author="ITCL",
    author_email="jorge.ruiz.itcl@gmail.com",
    description="A Simple Quantization Toolkit",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # packages=setuptools.find_namespace_packages(include=['itcl_quantization.*',]),
    # package_dir={
    #    "": "inference_engine",
    # },
    # py_modules=["itcl_quantization_inference_engine"],
    packages=["itcl_inference_engine"],
    install_requires=["numpy"],
)
