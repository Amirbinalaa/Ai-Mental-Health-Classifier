from setuptools import setup, find_packages

setup(
    name="ai-mental-health-monitor",
    version="0.1.0",
    description="AI-powered system to detect and monitor early signs of mental health concerns",
    author="AI Mental Health Monitor Team",
    author_email="contact@aimentalhealthmonitor.com",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    python_requires=">=3.9",
    install_requires=[
        # Dependencies are listed in requirements.txt
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)