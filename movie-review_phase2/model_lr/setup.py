#nsml: floydhub/tensorflow:1.7.0-gpu.cuda9cudnn7-py3_aws.27

from distutils.core import setup
setup(
    name='nsml Movie Reviews',
    version='1.0',
    description='',
    install_requires =[
	'keras',
	'jpype-py3',
	'konlpy',
	'lightgbm',
    ]
)
