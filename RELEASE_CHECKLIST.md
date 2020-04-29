# Release checklist

- [ ] Edit HISTORY.txt to make sure it’s up-to-date and add release date

- [ ] Edit the ‘What’s new?” section of the documentation to summarize important changes for users

- [ ] Remove previous version’s **NEW IN vX.X**

- [ ] Update the version number with ``bumpversion release`` (using [bump2version](https://github.com/c4urself/bump2version)) and commit.

- [ ] Check whether any new files need to go in MANIFEST.in (check which files are added with ``git diff --name-status PREV_RELEASE_HASH | grep ^A``)

- [ ] Make sure everything is committed and pushed, make sure tests run and pass

- [ ] Build source distribution: ``rm -rf build && rm -rf dist/* && python setup.py sdist``

- [ ] Push it to testpypi: ``twine upload -r pypitest dist/*`` and can test with ``pip install -i https://testpypi.python.org/pypi galpy``

- [ ] Tag new version with ``git tag vVERSION``

- [ ] Push new tag with ``git push --tags``

- [ ] Create new release on GitHub for this tag, with all of the links. Creating the release on GitHub will automatically build the source distribution and binary wheels using GitHub Actions and upload these to PyPI!

- [ ] Create the new conda builds at conda-forge —> now done automatically by bot, but still need to check that builds run correctly (should start within about half an hour from pushing the new release to PyPI)

- [ ] Switch default readthedocs version to the latest version

- [ ] Create maintenance branch if major version update

- [ ] Bump version to development version with ``bumpversion patch`` (for X.Y.1 --> X.Y.2) or ``bumpversion minor`` (for X.1.0 --> X.2.0).

- [ ] Start on next version changes in HISTORY file
