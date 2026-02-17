Step-by-step guide to quickly start a new project

- [ ] In github, clone the template directory and give it a meaningful name.
- [ ] clone this repository to your machine: copy the github repository url, open a git bash in the target location of your PC and type `git clone <url>`
- [ ] fill in the readme.md file.
- [ ] fill in the first line of license.txt with current year.
- [ ] update the requirements.txt file to contain all necessary dependencies.
- [ ] create a VS code workspace .code.workspace file to quickly open the workspace from windows: click on `file/save Workspace as...`
- [ ] create a virtual environment: use `ctrl+shift+p` and search for "Python: Create Environment...". 
- [ ] If not done automatically, activate the environment: `ctrl+shift+p` and search for "Python: Select interpreter". Open a command prompt (ctrl+`) to verify that the virtual environment is active. 
- [ ] If not done automatically, install the dependencies listed in requirements.txt by typing in a command prompt terminal: pip install -r requirements.txt.
- [ ] Install the helper_functions package as developer: in a command prompt (venv must be active), `pip install -e helper_functions`
- [ ] run "scripts/test_script.py" to make sure that the dependencies, including the custom package, are installed.
- [ ] stage and commit first changed to github (`ctrl+shift+g`), +, commit and sync. 
- [ ] create a data folder and import data if necessary.
- [ ] delete this file.