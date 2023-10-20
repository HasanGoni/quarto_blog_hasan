# Livecoding 1

* Try to remove all the python. So that we can start from the scratch
* use ``wget`` to get things from internet. wget the link of the mambaforge python link
* Then write ``bash <downloaded_file>``. This will install mamba. Now just anything works with conda, will work with mamba. mamba is similar like conda, but a little bit faster.
* ``mamba instll ipython``, which we need. 
* Now we need to close the terminal with ``ctl+d``. Then after opening we will see there is called (base)before computer name in the terminal. That shows you are in base environment. What actually happens, it is initialized in .bashrc file. If we open .bashrc file. we cann see that part. 
* Jeremy wants always the same initialization. So he created a script for that. So we will delete the part of .bashrc file, where mamba is initialized. Then we will go to fastai/fastsetup. There will be a file called ``set-up conda.sh``. Then we need to go to file, select raw at the right side. Then from the browswer we need to copy the link. And we will use our wget to get the file. so wget<link of the file>. This will downlaod the file. Now we need to run the file. 
* First we need to change the file, so that we can execute it. We will do it with the following command ``chmod u+x <file_name>``. Now run ``./<file_name>.
* This should run the file. Then it will initialize and we need to restart the terminal with ``ctl+d``.
* Now we will see again (base).
* Then ``mamba install juypterlab`` and ``mamba install ipywidgets``. To get pytorch we need to go pytorch get started and the select the necessary configuration. Then we get the command. Write the command. Instead of conda just write mamba. That should work.
* control r = reverse search, after pressing control +r type the things you are searching
* !ju will run the last command which start with ju
* !! will run the last command again
* echo !! will print the last command
* To go first of the line ctl+a
* To go end of then line ctl+e
* to go forward and backward for each word alt+f and backward alt+b

# Live Coding 2

* In linux we could have different user. We can change use by typing ``sudo -u <username> -i``. Here u is for user and i means we need an interactive terminal
* Create a git repo, try to select .gitignore for python
* Try to add license for Apache. Add also a readme file.
* In git you save different version of the file and also you can go anytime with your old version.
* In git saving named as commit, where you need to give some description(called as commit message) to the changes. In github there shows the commit and if you click the commit, then you can see the previous version of the file
* Normally we don't edit any thing is github, we need to have a version of the file locally, We need to download those files locally. In git grammar it is called clone. In github, there will be button called clone. There are differnt version of the cloning, we will select ``ssh``. 
* Then create a folder called git. Go to this folder. Type ``git clone and the link you got from github``
* Unfortunately that will not work. 
* Before goes any further, We need to tell the name and email address to our computer. 
* we can add two line ``git config user.name="hasan"``. ``git config user.email="hasan_test@email"``.
* Or we can just type ``cd``. This will move me in my home directory. There is a hiddenfile called .gitconfig. If we type ``ls -al`` we will see it. We can just edit ``.gitconfig`` file. about name and email address.
* Another useful command. I came to my home directory. But I now want to go the last directory I was in. ``cd -`` will move you to your last folder.
* One thing is problem here is maybe you want to go in a directory where you were earlier for a some time ago. you can try ``pushd`` and then after sometime ``pulld`` you can return the same directory, when you have done pushd
* Coming to github again. ssh don't need password, it works on keys. It generates a key for private and public. Anyone have the public key will be able to use your filesystem. That means you need to give them permission.
* To genearate a key for ssh, just go to command line. type ``ssh-keygen`` and type enter when asked. In your /home/<username>/.ssh/id_rsa there will two keys, one ends .pub, that will be pulblic. We need to tell github, about our public keys. 
* You can open the public keys in an editor. Or in linux, if ``cat /home/<user>/.sss/.id_rsa.pub will print the keys at your terminal. Copy the key.
* Next go to githbub and click the picture of you, click on settings.Then at the right side, there will ssh and gpg keys. Add ssh keys. Give a name so that you know about it. 
* Now if ``git clone <github ssh link>`` should clone the directory
* Now go to your by ``cd <repo name>. We can open jupyter lab --no-browser``. In windows no-browser is not necessary. But we are in linux. It tries to open the browser. Theerefore --no-browser is added here. Now we can not open use our terminal. We can open another terminal. But actually we can use another program called tmux(terminal multiplexer) to have different terminal sametime.
* just type tmux in another terminal. It should open tmux. If it is not there type ``sudo apt install tmux``. May be it will ask to you, your computer password. Here sudo means working as an administrator. 
* after installing( if not installed) type tmux. Then you will see the same thing. Just additionally a green bar will be available. You can use it simply like terminal all the command. But now if you press ``ctl+b %`` there will be a vertical  split of the terminal. And if you press ``ctl+b "`` it will create horizontal split of the terminal. To go different split, just press ``ctl+b <arrowkey>``
* If you are in a pane in tmux and want to zoom it. Means only to see the pane. press ``ctl+b z``, then you only see that pane. Again presss ``ctl+b z`` you will see your old view (all panes)
* Onething it is extreme useful in tmux. For example you were in a place and something is running there. You want it running. There you can just detach the tmux. And you can logout from there. detach is done ``ctl +b d`` and now need to attach ``tmux a`` will bring you the same place you wer in.
* One thing jeremy recommend to learn keyboard shorcuts. in Jupyterlab just type h, it should show some of the shorcuts. 
* Ok Now we can create a notebook in our repo. To add the file ``git add <filename>>``. In our case ``git add Untitle.ipynb``. Now we need to commit ``git commit -m "<any commit message>"``. Then ``git push``. This will push the notebook to our repo.
* Now we will install fastai. ``mamba install -c fastchan fastai`` should install the dependencies. Additioanly we need fastbook and sentencepiece. ``mamba install -c fastchan fastbook sentencepiece`` should install all the dependencies.
* Now we can clone the fastbook. Do git clone to locally. ``git clone -am "message". -a means all the changes and m is the message for this commit. 


# Live coding 3

* To create a virtual environment in mamba is similar like conda
* mamba create -n <name of environment><python version>. For example I want to create environment named as test_env with python version 3.7. Then the command will be ``mamba create -n test_env python=3.7 
* Normally the environment will be found /home/<user>/mambaforge/envs
* To activate the environment ``mamba activate <env_name>``. In our case ``mamba acttivate test_env``
* Paperspace using -->
* In jupyter lab right side, there is debugger.
* We can use magic function. Very first of the cell use %%debug in normal notbook environment. If we want to debug a function. At first line we can press ``s`` to tell please step into the funtion. We can alwas then press ``n`` to tell go to next line. If we press ``h`` it will show all the available key option 
* Or there is another option. For example ``from pdb import trace``. Then where we want to set a break point, we just tell there ``set_trace()``
* Another useful command for command ``df -h``, d stands for disk free  and h is human readable. This will show available spaces.
* In paperspace normally the libraries(e.g. fastai) will be saved in place called storage. ``storage`` folder is responsible for the packages.
* We can use another library with pip. Just pip install -U --user <library_name>. It will install the package in users home directory. Normally for simple libraries pip is ok, but where there are libararies, which is very much depedency with others, then it is good to use conda or mamba.
* We put --user and it will put the thing to .local folder.
* Now we want when we will add start our server again, we want this ``.local`` folder there. 
* We can actually move our ``.local`` to ``storage`` by ``mv .local storage/``. But we need .local to our home directory. What we can do is to create a symbolic link by ``ln-s``. ``ln -s storage/.local .``. What we are telling that, actually it will be a folder, but it is nor here, it is just pointing a place, where it is there. You can do everything with the folder, if it was there. 
* Paperspace has another ~/.bash.local folder, what actually you are telling. Whenever you start paperspace, first run those things for me. This way you can tell please do everything what is necessary for you(like we created symlink)
* In paperspace ~/.bashrc.local only run when you run the terminal

# Livecoding 4

* To delete everything before the cursor ``ctl+u`` and after the cursor ``ctl+k``
* To write same thing what we write last time is ``!$``
* Normally when we create a bash script we need to tell tell we want to execute it like that ``chmod u+x filename``. But Jeremy prefer to do it like ``chmod 755 filename`` 7-> user, 5->group 5-> everybody 
  * 1-> executable
  * 2-> writable
  * 3-> readable
  * 744->readable ,writable and exetable for user, 4 means readble for user, 4 means readable for groups
  * 700 --> user everything and no permission to anybody else
  * 600 --> only readable and writeble
  * 644 --> readable and writeable by me and only readable to everybody else
* if go to .ssh folder and try following command ``ssh git@github.com``, then no problem should occur to connect with github
* If that does not work. Try to see the verbose by adding v in this command. More v means more verbose. ``ssh -vvv git@github.com``
