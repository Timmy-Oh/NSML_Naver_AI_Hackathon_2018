#nsml: floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17

from distutils.core import setup
setup(
    name='nsml LSTM kin query',
    version='1.0',
    description='',
    install_requires =[
        'JPype1-py3',
        'konlpy',
        'progressbar2',
        'gensim'
    ]
)
