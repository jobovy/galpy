# Release checklist

- [ ] Edit HISTORY.txt to make sure it’s up-to-date and add release date

- [ ] Edit the ‘What’s new?” section of the documentation to summarize important changes for users

- [ ] Remove previous version’s **NEW IN vX.X**

- [ ] Update the version number in ``galpy/__init__.py``, ``setup.py``, and ``doc/source/conf.py``

- [ ] Check whether any new files need to go in MANIFEST.in

- [ ] Make sure everything is committed and pushed, make sure tests run and pass

- [ ] Tag new version with ``git tag vVERSION``

- [ ] Push new tag with ``git push --tags``

- [ ] Create new release on GitHub for this tag, with all of the links

- [ ] Before building source distribution, make sure ‘galpy/actionAngle/actionAngleTorus_c_ext/torus/‘ is not included in the current directory; best to start with a clean checkout!

- [ ] Build source distribution: ``rm -rf build && python setup.py sdist``

- [ ] Push to testpypi: ``twine upload -r pypitest dist/*`` and can test with ``pip install -i https://testpypi.python.org/pypi galpy``

- [ ] Push to pypi: ``twine upload -r pypi dist/*``

- [ ] Build wheels for different python versions: ``python setup.py bdist_wheel`` and upload with ``twine upload -r pypi dist/*.whl``

- [ ] Create wheels for different python versions on other platforms and upload with ``twine upload dist/*.whl`` (note that Linux wheels are not supported)

- [ ] Create the new conda builds at conda-forge —> now done automatically by bot, but still need to check that builds run correctly (should start within about half an hour from pushing the new release to PyPI)

- [ ] Bump master version to next X.dev version

- [ ] Start on next version changes in HISTORY file
