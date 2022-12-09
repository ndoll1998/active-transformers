from setuptools import setup, find_packages

setup(
    name="active-transformers",
    version="0.2.1",
    description="Active Learning for Transformer with focus on Sequence Tagging tasks",
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    license="Apache 2.0",
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: Freely Distributable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
    ],

    author="Niclas Doll",
    author_email="niclas@amazonis.net",
    url="https://github.com/ndoll1998/active-transformers/tree/master",

    packages=find_packages(exclude=["tests"]),
    package_dir={
        "active": "active",
    },

    python_requires=">=3.9",
    install_requires=[
        "torch>=1.11.0",
        "pytorch-ignite>=0.4.9",
        "transformers>=4.19.2",
        "datasets>=2.7.1",
        "scikit-learn>=1.1.1",
        "scipy>=1.9.0",
        "matplotlib>=3.5.2",
        "seqeval>=1.2.2",
        "pydantic>=1.10.2",
        "ray[default]>=2.1.0"
    ],
    extras_require={
        "rl": ["ray[rllib]>=2.1.0"]
    },

    tests_require=["pytest>=7.1.2"],
    entry_points={
        "console_scripts": [
            "active.train = active.scripts.run_train:main",
            "active.active = active.scripts.run_active:main",
            "active.rl = active.scripts.run_rl_active:main"
        ]
    }
) 
