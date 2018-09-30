from setuptools import setup, find_packages

requirements = [l.strip() for l in open('requirements.txt').readlines()]

with open('README.rst') as file:
    long_description = file.read()

config = {
    'name': '${python_package}',
    'description': '${description}',
    'long_description': long_description,
    'version': '${python_version}',
    'author': '${author}',
    'author_email': '${author_email}',
    'url': '${source_url}',
    'license': 'MIT',
    'include_package_data': True,
    'packages': find_packages(),
    'install_requires':requirements
}

setup(**config)
