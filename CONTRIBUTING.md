# How to create a pull request?

1. <a href="https://github.com/MLblog/jads_kaggle/fork">Fork the Gensim repository</a>
2. Clone your fork: `git clone https://github.com/<YOUR_GITHUB_USERNAME>/jads_kaggle.git`
3. Create a new branch based on `master`: `git checkout -b my-feature master`. The branch name should explain what functionality it is supposed to add or modify.
4. Setup your Python enviroment using the latest anaconda3 version, see notes below if you uncomfortable with this step.
5. Implement your changes
6. Check that everything's OK in your branch:
   - Check it for PEP8: `flake8 <path_to_source_folder/>`.
   - Run unit tests if any.
7. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature`
8. [Create a PR](https://help.github.com/articles/creating-a-pull-request/) on Github. Write a **clear description** for your PR, including all the context and relevant information, such as:
   - The issue that you fixed or functionality you added, e.g. `Fixes #123` or `Adds plots in EDA`
   - Motivation: why did you create this PR? What functionality did you set out to improve? What was the problem + an overview of how you fixed it? Whom does it affect and how should people use it?
   - Any other useful information: links to other related Github or mailing list issues and discussions, benchmark graphs, academic papers…

   
# Notes on development best practices

It is very important to make sure we all use the same development environment in order to manage dependencies without conflicts.
For example if contributor A pushes a new classifier using some libraries installed on his local machine, then the dependencies will not be met 
by other contributors after pulling, thus breaking their local copy. In order to ensure an isolated environment we will use `virtualenv`s. 
Here are the necessary steps on a windows machine:

1. Make sure you have a relatively clean <a href="https://www.anaconda.com/download/#windows">Anaconda 3.6 installation</a>.
2. Create a new [virtual environment](https://virtualenv.pypa.io/en/stable/): `pip install virtualenv; virtualenv kaggle_env;` 
3. Activate the newly created environment
	- Linux: `source kaggle_env/bin/activate`
	- Windows: `. kaggle_env/Scripts/activate`
4. Install the project's dependencies using `pip install -r requirements.txt`
5. In case your changes required the installation of extra packages remember to update the `requirements.txt` file: `pip freeze > requirements.txt`. This way others can install them as well


**Thanks and let's learn as much as possible together!**