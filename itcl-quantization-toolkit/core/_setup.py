import setuptools

setuptools.setup(
    name="itcl_quantization-toolkit",
    version="0.0.5",
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
    py_modules=["itcl_quantization"],
    python_requires=">=3.7",
    install_requires=["numpy"],
)
