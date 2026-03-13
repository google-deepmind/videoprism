# Copyright 2026 VideoPrism Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for VideoPrism.

Install for development:

  pip intall -e . .[testing]
"""

import setuptools

# Get install requirements from the REQUIREMENTS file.
with open("requirements.txt") as fp:
  install_requires_core = fp.read().splitlines()

tests_require = [
    "chex",
    "pytest",
]

setuptools.setup(
    name="videoprism",
    version="1.0.0",
    description=(
        "VideoPrism: A Foundational Visual Encoder for Video Understanding."
    ),
    author="VideoPrism Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/google-deepmind/videoprism",
    license="Apache 2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires_core,
    tests_require=tests_require,
    extras_require={
        "testing": tests_require,
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: VideoPrism/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="VideoPrism",
)
