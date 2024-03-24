lang="Python"

if [ ! -d TopLists ]; then
  mkdir TopLists;
fi

# Collect 25K repos with at least 500 stars.
# NOTE: the GH API neither guarantees nor (remotely) achieves completeness or consistency, so the resulting set of repositories will be different on each run.
# NOTE: make sure to insert your GH API key into the gh_crawler.py file.
python3 gh_crawler.py $lang


# Clone repositories in parallel and extract all language-specific files.
cat 'TopLists/'$lang'-top-repos.txt' | xargs -P16 -n1 -I% bash clone_repo.sh % $lang


# Deduplicate code files.
python3 deduplicate.py
