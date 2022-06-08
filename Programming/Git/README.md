
# Branches
## push local changes to a new branch
```shell
git switch -c <new-branch>
git add .
git commit -m 'message'
git push -u origin <new-branch>
```

## deltet a branch
### locally
`git branch -d <branchname>`
### remotely
`git push -d <remote_name> <branchname>`
In most cases, `<remote_name>` will be `origin`.

## list all branches
`git branch` list all local branches, the one with `*` is the current branch;
`git branch -r` list all remote branches;
`git branch -a` list all local and remote branches;
`git branch -vv` list all local branches details with address(?), correponding remote branch and last commited message;
`git branch -rvv` list all remote branches details with address(?) and last commited message;
`git branch -avv` list all local and remote branches details;

## check current status
`git checkout`.