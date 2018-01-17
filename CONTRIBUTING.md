# How to create a pull request?

1. <a href="https://github.com/MLblog/jads_kaggle/fork">Fork the Gensim repository</a>
2. Clone your fork: `git clone https://github.com/<YOUR_GITHUB_USERNAME>/jads_kaggle.git`
3. Create a new branch based on `master`: `git checkout -b my-feature master`. The branch name should explain what functionality it is supposed to add or modify.
4. Setup your Python enviroment
   - Create a new [virtual environment](https://virtualenv.pypa.io/en/stable/): `pip install virtualenv; virtualenv kaggle_env; source kaggle_env/bin/activate`
   - Install any package you need inside the virtual environment. Make sure everyone is using the same Python distribution (for exampole Anaconda3, python2.7 etc.) 
5. Implement your changes
6. Check that everything's OK in your branch:
   - Check it for PEP8 (new lines where due, consistent variable naming etc). In the future this can be automated.
   - Run unit tests if any
7. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature`
8. [Create a PR](https://help.github.com/articles/creating-a-pull-request/) on Github. Write a **clear description** for your PR, including all the context and relevant information, such as:
   - The issue that you fixed or functionality you added, e.g. `Fixes #123` or `Adds plots in EDA`
   - Motivation: why did you create this PR? What functionality did you set out to improve? What was the problem + an overview of how you fixed it? Whom does it affect and how should people use it?
   - Any other useful information: links to other related Github or mailing list issues and discussions, benchmark graphs, academic papers…

**Thanks and let's learn as much as possible together!**