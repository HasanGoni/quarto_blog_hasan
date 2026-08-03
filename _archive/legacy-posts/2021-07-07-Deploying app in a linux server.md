* Spinnig up new Linux server and tightening its secuirity with SSH keys and firewalls
# Using Linode Deploying on Linux Server
* Cloud.linode.com
* Select operating system and Region (near to the app)
* Select plan
* select root password
* Select or not (Backup or Private IP)
* Create and wait for all configuration done
* Go to networking tab
   * SSH Access --> machine address IP address will be available
## Entering first on the Server
   * ssh root@ip-address will be the ssh command 
   * command line (bash shell in my case) copy the previous line ssh command
   * Ask root password, that was previously selected by us.
   * after entering the password you will be at linode server.  

 ## Setup the server 
 * _apt update && apt upgrade_
 * _hostnamectl set-hostname flask-server_ # actully setting hostname. In our case we have set the hostname as flask-server
 * Need to open a file with an editor. we will use Vim. So _vim /etc/hosts_. 
     * We will see there 127.0.0.1 localhost. After this line we will write the ip address of server machine and then enter tab and write host name. Example _45.33.123.214 flask-server_
     * Save the file 

### Adding new user 
* _adduser hasan_
* maybe ask for password and we can set the password  for this user.
* Now ask some information for the user. Optional
* Now need to add this user to add sudo group, sothat this user can run admin command.
   * _adduser hasan sudo_

### Logout from the server and login as this new user
* _logout_
* _ssh hasan@45.33.123.214_ 
* ask for a password. Need to enter this user password. Actually we need to write ssh username@ip_address of the server machine and password of this username
* Now we are and logged in as a new user

### SSH keybased authentication. Login without password
* with our home directory we wiil make a _.ssh_ directory. _mkdir .ssh_
* Now open a bash window (if available, otherwise putty or other) in your local computer. Insert this command _ssh-keygen -b 4096_. If available they will ask you to overwrite or not. If you want you can overwrite. If this is first time then just ask for passphrase. You can make it empty, however if you want more secuirity then you can enter a passphrase. Now we got two extra file. _id_rsa_ and _id_rsa.pub_(pub means public)
* We actully need to move this publich key to our server. _scp ~/.ssh/id_rsa.pub hasan@45.33.123.214:~/.ssh/authorized_keys_ (scp source desitination [in our case destination is username@ipaddress:filelocation in server])
* Now check in the server whether the file is available or not. _ls .ssh_ .
* Now change the permissions _sudo chmod 700 ~/.ssh/_ , then ask for the password.
* Again _sudo chmod 600 ~/.ssh/*_ . For all files in ssh directory 
* Now we should be capable of login wihtout password. Check it. try to logout. _exit_
* In local machine bash shell _ssh hasan@45.33.123.214_. Now no password should be asked.

### Disallow root login over ssh
* Now update ssh config file in server  
* sudo vim _/etc/ssh/sshd_config_ . A password will be asked (because of sudo)
* We need to change two values of this configuration file
* __PermitRootLogin no__
* __PasswordAuthentification no __
* This two values need to be changed like above. Now we need to restart ssh server. _sudo systemctl restart sshd_

### Deploy a firewall
* _sudo apt install ufw_. ufw --> uncomplicated firewall
* _sudo ufw default allow outgoing_
* _sudo ufw default allow incoming_
* Now we need to allow certain rules to allow somthing which is necessary for us (ssh,http or certain port)
* _sudo ufw allow ssh_
* _sudo ufw allow 5000_ (5000 port is the port where we have created our flask application. So we need to test before we want to deploy it)
* Before allowing port 80 we need to make sure everything is working properly.
* Now we need to enable everything what we just set.
* _sudo ufw allow 5000_. Maybe will ask command disrupt existing connection. We will allow it
* You can always see what is allowed and what is not allowed. _sudo ufw status_

## Get Application to the server
* Mutiple ways (git, ftp), but we are using command line for copy so we will use the same.
* Go to our local command line bash window. 
* If we are using virtual environment, we need to create a requirement.txt file for all our libraries and versions.

### Creating requirement.txt
* pip freeze > requirement.txt
* Move this requirement.txt from home directory to project directory (In our case it is flaskblog directory)

### Copy project to server
* _scp -r Desktop/Flask_Blog hasan@45.33.123.214:~/_ (r means recursive means all child folder also. then source destination here we are using ~ means home directory of server)
* Go to server and check whether the file is there or not
* _ls_ . It should be there

## Create Virtual Environment on server
* _sudo apt install python3-pip_
* _sudo apt install python3-venv_ (This will allow us to create virtual environment)
* _python3 -m venv Flask_Blog/venv_ (This should create a virtual environment). Here Flask_Blog is the directory where we want the virtual environment and venv is the name of the virtual environment
* Now check whether the virtual environment is created or not
* Now to activate it we need _source venv/bin/activate_
* Now to install the requirement.txt we need to run _pip install -r requirement.txt_ (after -r we need to set the path of our requirement.txt file, here we have in the same directory therefore only requirement.txt is used)

## Create Configuration file with all of our environment varaiable
* There are different was to do this. If python environment is activated then we can just run python in bash 
* os.environ.get('SECRET_KEY')
* os.environ.get('SQLALCHEMY_DATABASE_URI')
* os.environ.get('EMAIL_USER')
* os.environ.get('EMAIL_PASS')
* Now go to server and create a file _sudo touch /etc/config.json_
* Now edit the file _sudo vim /etc/config.json_
* We will create the json file
   * {
      "SECRET_KEY": "579......",
      "SQLALCHEMY_DATABASE_URI": "sqlite:///site.db",
      "EMAIL_USER": "hasanme@gmail.com",
      "EMAIL_PASS": "password of email"
      }
* Now save the file
* Now goto our application where config.py is available. Open it  
    ```python
     import json
     with open('/etc/config.json') as config_file:
         config = json.load(config_file)
     ```
     * replace os.environ.get to config.get
     * Now save the file

## Run the flask app inside the server
* Upto now we have used python run.py. However we want to deploy it to use 0.0.0 that means open to outside world. Therefore we want to run it using flask run.
* To run it flask run we need to create an environment variable
* To create environment variable _export FLASK_APP=run.py_ in the command line. Means we are creating an environment variable
* flask run --host=0.0.0.0
* Now we want use our ip address means server ip address. In our case we used _45.33.123.214_ and also port needs to used. So we will use _45.33.123.214/5000_. Now our app should be available
* Do some testing. Everything is working or not. Create some post, Delete some post, Edit some post, create new user, and so on.. 

## Run the application with nginx and gunicorn
* Now kill the app from command line. control x
* Now go to home directory. _cd_
* Now run ```bash
              sudo apt install nginx
              pip install gunicorn
              ```
* Make sure that you are still in our virtual environment. 
* Now need to change some configuration with nginx and gunicorn

### nginx and gunicorn configuration
* ngingx will be our webserver and it will handle the request, such as static file like that.
* It will not handle the python code, gunicorn will do that
* Now remove nginx configuration file
```bash
sudo rm /etc/nginx/sites-enabled/default
```
* Now we are goind to create a new file
```bash
sudo vim /etc/nginx/sites-enabled/flaskblog
```
* Now we will write something in the file
```vim
server {
         listen 80;
         server_name 45.33.123.214;

         location /static {
                   alias /home/hasan/Flask_Blog/flask_blog/static;   
         }
         
         location / {
                   proxy_pass http://localhost:8000;
                   include /etc/nginx/proxy_params;
                   proxy_redirect off;
         }
}
```
* Save this file. What here is doind is actually telling where the location of the static folder, in our case all our static files are in the folder where we told after alias name
* Second location we are saying that if we are going to our ip address this will happen. Here localhost is server, write now local host is running on 8000 port
* Some extra varible for that proxy is supplied by next two lines
* Although we have said that we are listening to port 80. But still we donot created it
```bash
sudo ufw allow http/tcp
```
* Since we are done with testing therefore will disallow the port 5000

```bash
sudo ufw delete allow 5000
```
```bash
sudo ufw enable
```
* So we want to restart nginx server

```bash
sudo systemctl restart nginx
```
* So nginx is done. Now we need to gunicorn to handle python code
```bash
gunicorn -w 3 run:app
```
-w means number of workers. Then run is the file where our application is situated here(run), then colon and then variable name of our application here (app)

In gunicorn website it is said  number of workers = (2 * number of cores) + 1
* In linux how many core is available to see.

```bash
nproc --all
```
* To run gunicorn one should go to the directory
```bash
cd Flask_Blog/
```
now run gunicorn
```bash
gunicorn -w 3 run:app
```
Now if we open we will see it is working

* Now we need something which will monitor gunicorn, that means if it crashes it will restart again and will handle different thing to run it continuosly

#### Monitor gunicorn
```bash
sudo apt install supervisor
```
Now we need to creae a configuration file for sudo
```bash
sudo vim /etc/supervisor/conf.d/flaskblog.conf
```
In the configuration file we will write following line
```vim
[program:flaskblog]
directory=/home/hasan/Flask_Blog
command=/home/hasan/Flask_Blog/venv/bin/gunicorn -w 3 run:app
user=hasan
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/flaskblog/flaskblog.err.log
stdout_logfile=/var/log/flaskblog/flaskblog.out.log
```
* Atfirst where the program is available, then the command to run. Normally our virtual environment is not activated, therefore we should use whole path where the program is available. Then previously described file:varialbe
and other configuration for running swiftly. Also logfile directory is added. We did not create it yet. But we will create them
* So create those logfile
```bash
sudo mkdir -p /var/log/flaskblog
```
```bash
sudo touch /var/log/flaskblog/flaskblog.err.log
```
```bash
sudo touch /var/log/flaskblog/flaskblog.out.log
```
Now we need to restart supervisor
```bash
sudo supervisorctl reload
```
* Check everything
* One needs to know nginx has some default thing which can be different from our test, like previously we handled the large picture , but now nginx will not allow it
* Normally you don't need to change such default. But if you want to change it
```bash
sudo vim /etc/nginx/nginx.conf
```
Now if we scrool down to http
* one can set any variable for example client_max_body_size 5M;
* Now after saving them one need to restart nginx
```bash
sudo systemctl restart nginx
```
* Now it should work
* Actually we have completely functional website
* However we have only ip address. We can also buy a domain name
* also how to add ssl certificate to have a https domain name
* Now we can completely delete the server. 
* go to linode. and go the application server. right settings menu and goto settings. At the bottom there is a button, which says delete linode

# Buy domain name 
* Need to register to domain name. 
* Domain registers -- corey using namechip
* First register and then search for domain name
  * Normally difficult to find if available then will show the name and price. Also avialble other thing with same name (.com, .net etc.)
  * There are some additional add-on with extra cost, one actually can also add them. However corey donot use that.
  * There is a free addon called __whoisguard__. Corey recommended to use that because it protects privacy.
* After paying for domain name, there is button called manage. Now we are ready to connect to our applciation.
* Goto server Normally for this blog it was Linode. In linode there is option called _Domains_. In the right side there is some documentation _Linode Docs DNS Manager_. This will be used to connect the domain name with the server. 
* In the documentation 
    * buy domain
    * go to domain page and manage there is a opiton called _NAMESERVERS_ and select custom _DNS_ and change the name from linode server. Normally nsl.linode.com and so on..
    * There was right click sign which will save the name 
    * We need to wait some time. 24 hours to 48 hours. 
    * During this waiting time we can do something in linode server in the meantime.
    * Add Domain zone and adding some basic DNS recor ds
    * In linode server go to domain page and right side there is a plus sign which will create a form. Domain --> myawesomeapp.com and there will be email address admin@myawesomeapp.com. Add some tags, can be empty. Click on create 
    * There will be different records. We will add a _AAAA_ Record. We need a hostname and ip address. Hostname will be www and ip will be ip address of our application. Now save them the other option was like it was.  
    * Now do reverse DNS
    * go to linode and go the application and select Networking and IPv4 right side there will be some dotted sign and if we click on that we will see the two options, View and Edit RDNS. we need to click on _Edit DNS_. There was something. We need to put our domain name, in our case www.myawesomeapp.com. save that, if there is error, may be we need to wait some time 
    * Now copy the domain name and put them in the server.
    * Test all the things in the app
    * if we remove www. from the app it is not working. If we need to work it without www. we need to go again in Linode. go to myawesomeapp.domain.
    * Add another __AAAA_ record. In the hostname set the hostname blank and ip address will be our app ip address. It also take some time. may be 10-30 minutes and in worst case one day

   # Enable HTTPS with a free SSL certificate/ TLS Certificate
    * Right now in our wegsite it is written it is not secure. We will user __Let's Encrypt__.
    * In Let's Encrypt website, there is link for differnt operating system. For shell script there is a link for __certbot__. We are using _nginx and ubuntu_. We need to select those options
    * Then there will be documentation for that.
    * Available code for Nginx and Ubuntu
    ``` bash
    sudo apt-get update
    sudo apt-get install software-properties-common
    sudo apt-get-repository universe
    sudo apt-get-repository ppa:certbot/certbot
    sudo apt-get update
    sudo apt-get install python-certbot-nginx
    ```
    * Now we need to change nginx configuration. Now server name is ipaddress, now we need to change to our domain name
    ```bash
    sudo vim /etc/nginx/sites-enabled/flaskblog 
    ```
    In server
    {
        listen 80;
        server_name www.awesomeapp.com;
    }
    * Now save the file and run the commmand
    

```sh
    sudo certbot --nginx
    sudo certbot renew --dry-run
```
after first command ask some question. Number 2 Ridirect to secure site.
* It will show some changes are done on configuration file
let us see what has it done
```bash
sudo vim /etc/nginx/sites-enabled/flaskblog

```
one can see that it will redirect to secure opiton of the file

* If we want to see __nginx__ we will see there is some issue. 
``` bash
nginx -t
```
* here __t__ means test the configuration
* Now if we use same command usind sudo, then there will be no issue
``` bash
sudo nginx -t
```
## Allow https trafic to firewall
``` bash
sudo ufw allow https
```
Now restart the server
``` bash
sudo systemctl restart nginx
```
## Automate certificate after sometime
* We will dryrun. Dryrun actually to test everything is working as expercted or not, it will simulate everything but don't do anything
```bash
sudo certbot renew --dry-run
```
* So we will do cron-job
```bash
sudo crontab -e
```
Now we need to choose an editor. we will select 2 for vim
 ```vim
 30 4 1 * * sudo certbot renew --quiet 
 ```
 * quiet just do quietly
 Now save the file
 # Crontab example and tutorial
 * Here 30[Minutes] 4[Hours] 1[Day] *[Month:every month] *[Day of week:every day of the week]
 0--> sunday
 6 --> saturday
 comman for multiple values. 0,6 means only sunday and saturday
 */10 * * * *--> means we want something to run every 10 minutes
 * for example if we want something to run every 10 minutes only from midinight to 5 a.m
 ``
 0 0-5 * * * 
 ``
 

if we want something to run every 30 Minutes in common business hours
```*/30 9-17 * * 1-5 echo 'hello' >> /tmp/test.text```

* For a specific user 
```crontab -u user2 -e command```

* For a root 
```sudo crontab -l```--> list all crontab for root

now with e command one can edit the command 
```crontab -r``` remove all crontab

__cronGuru__ is a website where you can test those cronjob