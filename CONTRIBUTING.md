# Contribution guide and learning objectives

The steps below explain the process required to succesfully contribute to the project by submitting a pull request, thus closely simulating a real production environment. As a result our learning is 2-fold:

- We learn to work based on modern practices.
- By applying ML in a competitive environment we obtain in depth knowledge of the topics and algorithms used.

Understanding certain steps in the process assumes a basic grasp of certain important topics, not all of which are trivial. At the end of this document, an ever expanding [list of learning resources](#learning-resources) will be provided. Contributors are strongly encouraged to make use of this list.

# How to create a pull request?

1. <a href="https://github.com/MLblog/jads_kaggle/fork">Fork the repository</a>
2. Clone your fork: `git clone https://github.com/<YOUR_GITHUB_USERNAME>/jads_kaggle.git`
3. Create a new branch based on `master`: `git checkout -b my-feature master`. The branch name should explain what functionality it is supposed to add or modify.
4. Setup your virtual Python enviroment using the latest `anaconda` version (currently `3.6`). Check the [Notes](#setting-up-a-virtual-environment) for how do to that.
   Make sure you activate the environment everytime you start working on the project.
5. Implement your changes.
6. Check that everything is OK in your branch:
   - Check it for PEP8 by running: `flake8`. This requires that a virtual environment containing the `flake8` library (like our `kaggle_env`) is activated.
   - Run unit tests if any: `pytest`.
7. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature` where `origin` is your own fork.
8. [Create a PR](https://help.github.com/articles/creating-a-pull-request/) on Github. Write a **clear description** for your PR, including all the context and relevant information, such as:
   - The issue that you fixed or functionality you added, e.g. `Fixes bug with...` or `Adds plots in EDA`
   - Motivation: why did you create this PR? What functionality did you set out to improve? What was the problem + an overview of how you fixed it? Whom does it affect and how should people use it?
   - Any other useful information: links to other related Github or mailing list issues and discussions, benchmark graphs, academic papers…

   
# Setting up a Virtual Environment

It is very important to make sure we all use the same development environment in order to manage dependencies without conflicts. For example if contributor A pushes a new classifier using some libraries installed on his local machine, then the dependencies will not be met by other contributors after pulling, thus breaking their local copy. In order to ensure an isolated environment we will use `virtualenv`s. 
Here are the necessary steps to create and activate one on a windows machine:

1. Make sure you have a relatively clean <a href="https://www.anaconda.com/download/#windows">Anaconda installation</a>.
2. Create a new [conda virtual environment](https://conda.io/docs/user-guide/tasks/manage-environments.html): `conda create --name kaggle_env --file requirements.txt;` 
3. Activate the newly created environment: `source activate kaggle_env`. This step should be repeated everytime you start working on the project.
4. In case you are using an IDE like Pycharm or Spyder, make sure it uses your conda environment as the project interpreter.
	- Your environment will be by default located at: `<Anaconda Home>/envs/kaggle_env/python.exe`
	- Default location for `<Anaconda Home>` on Windows is `C:\Users\<username>\AppData\Local\Continuum\anaconda3\`
5. In case your changes required the installation of extra packages (for example via `conda install <package>` remember to update the `requirements.txt` file: `conda list --explicit > requirements.txt`. 
   This way others can install them with `conda install --file requirements.txt`
6. You can leave the virtual enviroment at any time using `source deactivate`
## Task 0 - Getting comfortable with pull requests ##

In order to test your understanding of the proposed process you can try a rather minimal contribution:
Follow the guidelines provided above and make a pull request to add your name in the contributors list found in `README.md`

# Pulling changes performed by others

*As we are working on the project, others might be doing the same. In order to be able to benefit from their work we need a way to access it in our copy of the codebase, which is easy in Git using `git pull`.*


Others are obviously submitting pull requests on the original repository, not in our fork. Our local copy is by default only connected with our own fork because this is where we cloned from. 
Online repositories are called `remote`s and the default `remote` is called `origin`. Therefore if we now check the connected remotes by running `git remote -v`, we will only see our own fork named `origin`.
In order to connect our local copy with the original remote as well, we can issue `git remote add base https://github.com/MLblog/jads_kaggle.git` where `base` is the name we select for this new `remote`.
From now on everytime we want to `push` or `pull` changes we can do so by specifying the `remote` and branch. For example in order to get changes performed by others we can do: `git pull base master`.
In order to `push` our branch called `my-feature` to our fork we can run `git push origin my-feature`. Please note that you don't have permission to `push` to `base` (try and see what happens), 
instead you need to `push` into `origin` and submit a PR as explained above.

# Resolving Conflicts

*ToDo*

## Learning Resources
The aforementioned process assumes a basic understanding of certain software tools. The list below will serve as a reference for present and future contributors of the project.

1. Git
    - [Basics](https://guides.github.com/activities/hello-world/)
    - [Branching](https://guides.github.com/introduction/flow/)
    - [Forking](https://guides.github.com/activities/forking/)
    - [Cheat sheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf)
    - [Udacity Course](https://eu.udacity.com/course/how-to-use-git-and-github--ud775) (in case you want a deeper understanding - strongly recommended)

2. Virtual Environments
    - [Managing environments in Anaconda](https://conda.io/docs/user-guide/getting-started.html#managing-envs)
    - [Cheat sheet](https://conda.io/docs/user-guide/cheatsheet.html)

3. Object Oriented Programming
    - [Classes](https://docs.python.org/3/tutorial/classes.html) in Python. Extensive tutorial containing valuable information.
    - [Abstract Classes](https://docs.python.org/3/library/abc.html) A more complex but also powerful concept used in the project.

4. Unit testing 
    - [PyTest](https://docs.pytest.org/en/latest/contents.html#toc)
    - [Travis](https://stackoverflow.com/questions/22587148/trying-to-understand-what-travis-ci-does-and-when-it-should-be-used): A continuous integration tool we will be using.

**Thanks and let's learn as much as possible together!** 

