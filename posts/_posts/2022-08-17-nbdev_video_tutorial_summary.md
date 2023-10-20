# Nbdev worker round video tutorial

-author: Hasan

* This is the summary of [this](https://www.youtube.com/watch?v=67FdzLSt4aA&ab_channel=JeremyHoward)video.

## Create a repo in Github
* Create an empty repo.
* clone it to local place
## Install nbdev
* visit homepage how to install 
* pip or conda
* after installing try to write ``nbdev_help``,you will see all available command and small description what the function do.
## Working with nbdev 

* Now try ``nbdev_new``
* This will create a file called settings.ini 
* This will be home and create all necessary information for your repo
* index.ipynb will be the homepage for the docs
* 00_core.ipynb will be the place where you create your library
* Actually we need two modules cards and decks, So here jeremy rename 00_core.ipynb to 00_cards.ipynb and 01_decks.ipynb
* Each module wil create a pyfile 
* Jeremy change the very first comment default_exp core to default_exp card
* There are some comments which just starts with # and then pipe, like ``#|`` hide means this will be hidden in your doucmentation
* ``default_exp is called default export
* It is a good idea to see a library which is written in nbdev and same time see the rendered documentation
* After creating function or class, we can write docstring and it will be used as documentation. We can see use ``show_doc(function_name)`` to see the preview
* We can actually comment each parameter and some information regarding this
* Normally in same notebook showdoc is not necessary. But may be some other function which is not created using nbdev, but we need that code here and we want to document it, therefore we can use show_doc then
* from fastcore.test import *
* test_eq(function_result, expected_result), test_ne(function_result, any_result)
* outside of a class define whether equal. from fastcore.utils import *, use decorator @patch def __eq__(self:Card, a:Card) return (self.suit, self.rank) == (a.suit, a.rank)
* Section can be introduced using 2 heading(##)
* We can also use __lt__ using patch or __gt__ . To test less than or greater than use assert or not assert
* When first card is done, go to terminal try to nbdev_export. After that try pip install -e .
* Now we will see nbdev_card directory with card.py file
* Now go to index, try to import it and see everything works
* To install it pip install nbdev-card or conda install -c fastai nbdev-cards
* change setting.ini. lib_name:nbdev-card, lib_path:nbdev_cards
* nbdev_preview to preview the documentation-> this will fire quarto webserver , install quatro, if you don't have it
* now again run again nbdev_preview. local_host:3000, there you will see the documentation
* nbdev_test in terminal it will tell you whether all test successful or not 
* pip install -e . means editable, it will  pointer to the github repo
* Try not to mix import and other cells in a same cell. So in a import cell only import should be available other things should not be there
* nbdev_clean -> clean all unnessary metadata from notebooks
* git diff -> see difference
* nbdev_export
* nbdev_test
* nbdev_prepare(nbdev_export, nbdev_test, nbdev_clean)-> together
* after pushing to github, go to settings->pages->Branch->gh-pages->save
* go to actions->pages_build_and_deployment->when deployment is done->copy the link
* go to code and settings->includ in home page->uncheck packages->uncheck Environments and in website paste the links
* nbdev_docs  -> create same readme like in github pages
* nbdev_pypi -> will publish the pip intall package
