# Create docker image from docker file, run it and push to ibm registry 

## Introductory work and hello-world running
* First check the docker cli is installed 
  * that can be done by following command `docker --version`
* Check ibmcloud cli is installed or not.
  `imbcloud version` will tell you whether ibmcloud is installed or not
* Change to project folder
  * `cd /home/project`
  * `[ ! -d 'CC201' ] && git clone https://github.com/ibm-developer-skills-network/CC201.git`
  * `cd CC201/labs/1_ContainersAndDocker/`
  * see available images `docker images`
  * pull first dokcer image from docker hub `docker run hello-world`
  * Now we can see all the running images by `docker ps -a`
  * Now we can remove the container by its container id `docker container rm <container id>`
  * lets check whether it still running or not `docker ps`

## Build an image using docker file

* We already cloned the docker filea and app
* docker file is the following.
  
  ```bash
  
  FROM node:9.4.0-alpine
  COPY app.js .
  COPY package.json .
  RUN npm install &&\
    apk update &&\
    apk upgrade
  EXPOSE  8080
  CMD node app.js
  ```
* we can build the image by running ``dockcer build . -t myimage:v1``
* let see all the images ``docker images``

## Runing the image

 * Now that we have built the image, we can run it `docker run -dp 8080:8080 my_image:v1`
 * Now check whether app is running . `curl lacalhost:8080`, this will tell some output which confirms that the app is running

## Stop the image
 
 * Now we can stop it by running ``docker stop <container id>``
 * We can actually actually stop all the containers ``docker stop $(docker ps -q)``
 * docker ps will tell whether everything is stopped or something is running

## Push image to container registry

 * First need to logiin ibmcloud
 * then following command will tell information regarding the account `ibmcloud target`
 * Need to create an IBM cloud container Registry(ICR) namespace. Since Container Registry is multitenant, namespace will devide the registry among different users
 * ibmcloud cr namespaces
 * target the desired region, here in coursera the region is set us-south, ``ibmcloud cr region-set us-south``
 * Now need to login to account ``ibmcloud cr login``
 * export namespace as environment variable so that, it can be used later. ``export MY_NAMESPACE=sn-labs-$USERNAME``
 * Tag the image so that it can be pushed ``docker tag myimage:v1 us.icr.io/$MY_NAMESPACE/hello-world:1``
 * Now push the image `docker push us.icr.io/$MY_NAMESPACE/hello-world:1`
 * Verify the images are available or not  ``ibmcloud cr images``
 * If we want to see only the image availabe we can see it ``imbcloud cr images --restrict $MY_NAMESPACE``

## Delete the project 
 
 * From Open shift console goto project -> click action->delete project

