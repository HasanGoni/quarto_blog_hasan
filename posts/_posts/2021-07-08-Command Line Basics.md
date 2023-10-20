* This tutorial is taken from Corey schafer youtube video


* At first we need to configure the global configuration. That means who is working with the projects
* `` git config --global user.name "HasanGoni"``
* `` git config --global user.email "hasanme1412@gmail.com"``
* If we want to see all the configuration in this computer we can usually `` git config --list``
* `` git help <verb> ``
# Two scenario to use git 
* Local machine existing project you want to track it 
* Existing project remotely you want develope.

## 1. Local project

### Initializing the project
``git init``
* If we want to stop tracking the project we need to remove the .git directory. ``rm -r .git``

### Observe the status of the project
* ``git status`` to see the status of the project

### Ignoring certain files
* To ignore some files need to add ``.gitignore`` file in the directory. ``vim .gitignore`` 

* For example in python we don't want to track pycache files. So we want to add ``.DS_Store`` in ``.project`` and ``*.pyc``. Here we are saying please ignore .project directory and .DS_Store directory and ignore all files which ends with .pyc file extension.

### Adding files
* We will add those lines to the .gitignore files and save them
* Adding files, we can just add all untracked files with `` git add -A``
* Or we can just tell please add all the files. `` git add``
* Or one can add just a single file like ``git add .gitignore``. Here we just adding a single .gitignore file.

### Remove files from staging area
* If we want to remove a file from staging area we can just say `` git reset <filename> ``. e.g. ``git reset .gitignore``
* Like previously if we want everything to remove from staging area we can just say ``git reset``

### Commiting
`` git commit -m "Explanatory message of the change"``
* if we want to see the status, we can use `` git log``, which will show status and hashed history with commit message.

# 2. Track existing remote project with git

### Clone remote repository
`` git clone <url> <where to clone>``
* e.g. `` git clone ../remote_repo.git . ``. here __.__ is current directory    

### Inoformation about remote repository 
* ``git remote -v`` --list information of the repositroy
* `` git branch -a`` --> shows all the branches in the repository. Remote and local
* Now after doing some change we need to push it
### Show information of the changes done
`` git diff``
`` git status`` shows what needs to be commited and what already commited
### Adding file
`` git add -A `` adds to staging 
`` git status`` shows again the status. May be this time will show everything already commited because of previously git add
### Commit
`` git commit -m "Information of the code" ``
* Now we want to push those changes to remote repository
### Pushing to remote repository	
* First need to do ``git pull``
* ``git push origin master`` --> origin is the name of the remote directory and master is the branch where we want to push

# Common Workflow
  * Create branch
  ``git branch new_branch``
  * Go to that branch
  ``git checkout new_branch``
  * Adding file
  `` git add -A``
  * Commit
  `` git commit -m "message Information"``
  * Pushing in Remote 
  `` git push name_of remote_directory name_of_local_branch``
  `` git push origin new_branch``
  * Show available branch
  `` git branch -a``
  * Checking out local master
  `` git checkout master``
  `` git pull origin master``
  * Check which branches are merged
  `` git branch --merged``
  * Merging with master
  `` git merge new_brach`` when you are in master branch 
  * Push to remote brach
  `` git push oringin master``
  * Check again all the merges
  `` git branch --merged``
  * Delete new creeated branch
  `` git branch -d new_branch``
  * Delete branch from remote repository
  `` git push origin --delete new_branch`` 

### More advanced tutorial can be found [here](https://hasangoni.github.io/2021/07/08/Fixing-Mistakes-And-Bad-Comments.html)