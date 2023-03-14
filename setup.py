from setuptools import setup, find_packages

setup(
    name="ImplicitNeuralRepr",
    version="0.1.0",
    author="Zhexin Wu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "SimpleITK",
        "nibabel",
        "pytorch_lightning",
        "monai",
        "PyYAML",
        "einops"
    ]
)
