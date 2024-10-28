import setuptools

with open("README.md","r",encoding = "utf-8") as f:
    DESC = f.read()

PCK_DESC = "Python packages for NetworkSecurity"
__version__="0.0.0"
SRC_REPO = "src"
REPO_NAME = "MLOps-NetworkSecurity"
AUTHOR_EMAIL = "tysonbarretto1991@gmail.com"
AUTHOR_NAME = "tysonbarreto"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=PCK_DESC,
    long_description=DESC,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls={
        "Bugs Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues"
    },
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where = "src")
)