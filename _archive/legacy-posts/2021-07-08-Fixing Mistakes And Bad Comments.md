source [Corey Schafer Tutorial](https://www.youtube.com/watch?v=FdZecVxzJbk&list=PL-osiE80TeTuRUfjRe54Eea17-YfnOOAx&index=2&ab_channel=CoreySchafer)

## Basic command line tutorial can be found [here](https://hasangoni.github.io/2021/07/08/Command-Line-Basics.html)
# Branch change
* For example we have made changes in a file. After some time, we have found that we no longer needed to those chnages. And we are in master branch. There is another branch available, which is not previously same as master branch. Now we have some changes in master branch. Therefore simple checkout can help us in this scenario. 

``git checkout another_branch``. Here another_branch was the branch which was previously identical to master branch.

# Wrong Commit Message 
* Second criterial, where we have done chnages in a file.
  * ``git add -A``
  * ```git status ``` wil show the file which was ready for commit. However we have made a bad commit. 
  * `` git commit -m " bad commit"``. How we will change the commit message without making another commit. ``git commit --amend -m " Again good commit"``. Now if we want to see the ``git log`` we will see that only Again good commit is available.
  * however we need to consider that this is only for local changes, if we already pushed our code outside world, then we should do it another way, becuase the hash already changed, which causes the history change in git. So for remote repository maybe it can cause error

  * Let say we accendently forgot to commit some file. For example we have added a .gitignore file. We have added to staging area. ``git add -A``. We want to commit at last. 
  If we do ``git commit --amend`` it will open all of our commits in editor, where we can change commit message if we want. We actually don't want to change something right now. Now if we do ``git log --stat`` we will see that, .gitignore file is actually now a part of the last commit. If we do ``git branch`` we will see that there in another branch. Actually we have done al our chages to master branch. Actually we wanted to do that in another branch. So we want to change our last commit to another_branch and master branch will be like previous

# Wrong branch Commit
    * ``git log`` we will pick the hash of last commit(full hash not necesarry 6/7 is enough). Copy this hash part.
    * ``git checkout another_branch`` here another_branch is the branch where we wanted to the commit. Now if we perform ``git log`` we will see that we have only one commit. Then we will do ``git cherry-pick copied_hash``.Now if we want to see the ``git log``. So we have our commit in our another_branch. Now we want to change the master branch. ``git checkout master``. Now we are in master branch. We have 3 types of reset.
               1. git reset --soft
               2. git reset --mix(default)
               3. git reset --hard
    So again we will copy the commit hash. We will do `` git reset --soft copied_hash``. Now ``git log`` shows that we no longer have the commit. However ``git status`` will show that we have some files in our staging area.
    If we perform mix reset. ``git reset copied_hash``. Agian ``git log`` will show that we have only one commit.``git status`` will show that we have files not in staging area but in working directory. Now in that case we actually donot want our files here we just want them to remove. ``git reset --hard copied_hash`` . __Be careful in real world application before reseting this__. 
    Now we if again see that ``git status`` there will be `.ignore` file still untracked. ``git reset --hard`` reverts all of the tracked files back to states they were, and left untracked files alone. Getting rid of untracked files ``git clean -df`` here d=directory, f=file. Now ``git status`` will show that we have our desired situation.

# Getting file after hard --reset (reflog)
* may be after 1 Month or so ..not sure how much time we have to get such back. One should try

* What happens when we have done git reset But we realized that some of our changes actually is needed. ``git reflog`` and copy the hash where we need the status.`` git checkout copied_hash`` Now if we perform ``git log`` then we can see the hash of our need. 
* Right now we are in __detach head__ state. Means it will trashed out in near future. To save chagnes we need a brach for it ``git brach backup``.``git checkout backup`` will be our brach which was necessary. 

# Correcting Commit after Pushing to remote (git revert)
* If we already made our mistakes. And pushed it to remote repository. some people already made their changes also.
* For example we have two commits. we have __normal commit__ and __bad commit__. This __bad commit__ is the commit we want to change, however already some changes are done by other users. We will copy the hash from ``git log`` the hash of __bad commit__. ``git revert copied_hash``. There will some message in our editor, we will save them. Now if we will see ``git log`` we will see that we have added another commit. The last commit will be ``reverted bad commit`` If we want to see the difference between two hashes, we can just ``git diff first_hash, second_hash``  