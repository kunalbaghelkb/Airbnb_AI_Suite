from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This Function will read requirements.txt file and it will return libraries list
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # \n to remove new lines
        requirements = [req.replace("\n", "") for req in requirements]
        
        # remove '-e .' from list it is not a library
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
        return requirements
    
setup(
    name='Airbnb_AI_Suite',
    version='1.0',
    author='Kunal Baghel',
    author_email='126225910+kunalbaghelkb@users.noreply.github.com',
    packages=find_packages(), # Automatically finds packages containing __init__.py
    install_requires=get_requirements('requirements.txt')
)