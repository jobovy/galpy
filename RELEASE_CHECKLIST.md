# Release checklist

- [ ] Edit HISTORY.txt to make sure it’s up-to-date and add release date

- [ ] If major version update, edit the ‘What’s new?” section of the documentation to summarize important changes for users i

- [ ] If major version update, remove previous version’s **NEW IN vX.X**

- [ ] Update the version number with ``bumpversion release`` (using [bump2version](https://github.com/c4urself/bump2version)) and commit.

- [ ] Check whether any new files need to go in MANIFEST.in (check which files are added with ``git diff --name-status PREV_RELEASE_HASH | grep ^A``)

- [ ] Make sure everything is committed and pushed, make sure tests run and pass

- [ ] (Optional check) Build source distribution: ``rm -rf build && rm -rf dist/* && pip install build && python -m build --sdist``

- [ ] (Optional check) Push it to testpypi: ``twine upload -r pypitest dist/*`` and can test with ``pip install -i https://testpypi.python.org/pypi galpy``

- [ ] Tag new version with ``git tag vVERSION``

- [ ] Push new tag with ``git push --tags``

- [ ] Create new release on GitHub for this tag, with all of the links and a brief summary of major updates. Creating the release on GitHub will automatically build the source distribution and binary wheels using GitHub Actions and upload these to PyPI!

- [ ] ~(Optional; currently not necessary) Create wheels using the ``create_galpy_wheels.sh`` script for platforms that aren't supported by the CI builds (all platforms are currently CI built). Upload these to PyPI with ``twine upload wheels_directory/*``~

- [ ] ~Create the new conda builds at conda-forge~ —> now done automatically by bot, but still need to check that builds run correctly (should start within about half an hour from pushing the new release to PyPI)

- [ ] Switch default readthedocs version to the latest version

- [ ] ~~If major version update, create maintenance branch~~

- [ ] Bump version to development version with ``bumpversion patch`` (for X.Y.1 --> X.Y.2) or ``bumpversion minor`` (for X.1.0 --> X.2.0).

- [ ] Start on next version changes in HISTORY file
