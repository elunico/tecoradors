#!/usr/bin/env zsh

echo -n 'Did you increment the version number (y/n)? '
read -q
echo

if [[ "$REPLY" == 'y' ]]; then
  rm -rf dist &&
  python3 setup.py sdist bdist_wheel &&
  twine upload dist/*
else
  echo 'Update the version number first'
  echo "Checking setup.py for version number"
  echo "|- $(cat setup.py | grep version)"
fi
