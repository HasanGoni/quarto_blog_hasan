# Scaling and Updating Applications

## Theory

* Config Maps:
	* A way to provide configuration data to pods and deployments; so, we they don’t need to be hard-coded in the application code. ConfigMaps can be reused for multiple deploymentsdecoupling the environment from the deployments themselves.
 
* HPA:
	* The “Horizontal Pod Autoscaler” enables an application to increase the number of pods based on traffic.

* Rolling Updates:
	* Provide a way to roll out application changes in an automated and controlled fashion throughout your pods. Rolling updates work with pod templates such as deployments. Rolling updates allow for rollback if something goes wrong.

* Secrets:
	* Work similarly to ConfigMaps but are meant for sensitive information.
	
## Objectives

* Scaling an application with ReplicaSet
* Apply rolling updates to an applications
* Using a ConfigMap to store application configuration

## Scaling an application using ReplicaSet

### Preparation work

* Clone github repo containing required information(app and files)
* `[ ! -d 'CC201' ] && git clone https://github.com/ibm-developer-skills-network/CC201.git`
* Change to directory for this lab `cd CC201/labs/3_K8sScaleAndUpdate` 

### Build and push image to ibmcloud container registry

* export the namespace as an environment variable
* `export MY_NAMESPACE=sn-labs-$USERNAME`
* The dockerfile content  
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
* Build and pushing image
  `docker build -t us.icr.io/$MY_NAMESPACE/hello-world:1 . && docker push us.icr.io/$MY_NAMESPACE/hello-world:1`

### Deploy the application to Kubernetes

* There will be a file named as deployment.yaml. The content of the file is following

```bash
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-world
spec:
  selector:
    matchLabels:
      run: hello-world
  template:
    metadata:
      labels:
        run: hello-world
    spec:
      containers:
      - name: hello-world
        image: us.icr.io/sn-labs-hasanme1412/hello-world:1
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: 2m
            memory: 30Mi
          requests:
            cpu: 1m
            memory: 10Mi  
        
      imagePullSecrets:
        - name: icr
```
* Run the image with the following  

```bash
kubectl apply -f deployment.yaml
```

* List pods to see whether they are running and wait until all pods are running, `kubectl get pods`
* Now we need to expose the app, `kubectl expose deployment/hello-world`
* Open a new terminal and and write `kubectl proxy`
* Go back actual terminal and run the following command. 
  ```bash
  curl -L localhost:8001/api/v1/namespaces/sn-labs-$USERNAME/services/hello-world/proxy
  ```

### Scaling an application

* Use scale command to scale up the deployment. Make sure run the following command which is **not** running in proxy.`kubectl scale deployment hello-world --replicas=3`
* Make sure whether they are running by `kubectl get pods`

### Scale down an app

* Scaling down can be done similar way, `kubectl scale deployment hello-world --replicas=1`
* Check again with `kubectl get pods`

## Rolling Updates

* For example if there is a change in your app. We can then build the new image and push it to registry with new tag.
* ```bash
docker build -t us.icr.io/$MY_NAMESPACE/hello-world:2 . && docker push us.icr.io/$MY_NAMESPACE/hello-world:2
* we can check whether the new image is available or not, `ibmcloud cr images`
* Update the deployment to use this version instead.`kubectl set image deployment/hello-world hello-world=us.icr.io/$MY_NAMESPACE/hello-world:2`
* Get a status of the rolling update by using the following command `kubectl rollout status deployment/hello-world`
* You can also get the Deployment with the wide option to see that the new tag is used for the image.`kubectl get deployments -o wide`
* Ping your application to ensure that the new welcome message is displayed.`curl -L localhost:8001/api/v1/namespaces/sn-labs-$USERNAME/services/hello-world/proxy` 
* It's possible that a new version of an application contains a bug. In that case, Kubernetes can roll back the Deployment like this:`kubectl rollout undo deployment/hello-world`
* Get a status of the rolling update by using the following command:`kubectl rollout status deployment/hello-world`

## Config Maps to stor Configuration

* Create a ConfigMap that contains a new message.`kubectl create configmap app-config --from-literal=MESSAGE="This message came from a ConfigMap!"`
* There is a file in this repo called deployment-configmap-env-var.yaml, containing following content
```bash
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-world
spec:
  selector:
    matchLabels:
      run: hello-world
  template:
    metadata:
      labels:
        run: hello-world
    spec:
      containers:
      - name: hello-world
        image: us.icr.io/<my_namespace>/hello-world:3
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: app-config
      imagePullSecrets:
        - name: icr
```

* In the app we can change the message to `process.env.MESSAGE`
* Now we can build the image and push it again.
```bash
docker build -t us.icr.io/$MY_NAMESPACE/hello-world:3 . && docker push us.icr.io/$MY_NAMESPACE/hello-world:3
```
* Now try the following command `kubectl apply -f deployment-configmap-env-var.yaml`
* Try to see the app. `curl -L localhost:8001/api/v1/namespaces/sn-labs-$USERNAME/services/hello-world/proxy`
* Because the configuration is separate from the code, the message can be changed without rebuilding the image. Using the following command, delete the old ConfigMap and create a new one with the same name but a different message.`Because the configuration is separate from the code, the message can be changed without rebuilding the image. Using the following command, delete the old ConfigMap and create a new one with the same name but a different message.`
* Restart the Deployment so that the containers restart. This is necessary since the environment variables are set at start time.
* `kubectl rollout restart deployment hello-world`
* ping the app ``curl -L localhost:8001/api/v1/namespaces/sn-labs-$USERNAME/services/hello-world/proxy``
* Delete the deployment `kubectl delete -f deployment-configmap-env-var.yaml`
* ctl+c to stop proxy
