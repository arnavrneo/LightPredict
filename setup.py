from setuptools import setup

setup(name="lightpredict",
version="0.05",
description="Test 1",
long_description = "This is a very very long description",
author="Arnav Raina",
author_email="arnavr.neo@gmail.com",
url="https://github.com/arnavrneo/LightPredict",
keywords = ['machine learning', 'scikit learn', 'sklearn', 'ai', 'optuna', 'hyperparameters'],
packages=['lightpredict'],
install_requires=[
	'numpy',
	'pandas',
	'matplotlib',
	'sklearn',
	'optuna'
])