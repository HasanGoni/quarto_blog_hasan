# Using Kubectl to create deployment

- A very nice and detailed description can be found [here](https://kubernetes.io/docs/tutorials/kubernetes-basics/deploy-app/deploy-intro/)
- I am just summarising it here.
- We need **Deployment** configuration
- This **Deplyment** how to create and update the application
- Once the application instances are created, **Kubernetes Deployment controller** creates a self healing controller for this application. In case of any nodes have problem, it replaces it and create a new one.
- _kubectl_ is command line tool with which we can perform our job(update, create or get information)

## Command line

- `kubectl version` > to see the version of the kubectl.
- `kubectl get nodes` > to see the informaiton about the nodes.
- `kubectl create deployment kubernetes-bootcamp --image=gcr.io/google-samples/kubernetes-bootcamp:v1` create deployment. We need to tell deployment name and image location.
- `kubectl get deployments` to see all available deployments in use.
- Right now these pods are only visible to kubernetes cluster. We will see later how can these pods will be visible to outside kubernetes.
- We can create a proxy `kubectl proxy`, now we have a connection between our host and kubernetes cluster.
- We can check it with `curl http://localhost:8001/version`

# Viewing pods and nodes

- kubernetes pods are abstraction, that represents group of containers. Some shared resources for those containers

  - Shared storages such volumes
  - Networking, as unique IP adress
  - Information about how to run each container. such container image version and specific port to use.

- Nodes are virtual or physical machine which run these pods.
- Nodes run atleast:
  - kubelet > communication between kubernetes `control plane` and Nodes. kubletes manages Pods and container running on the machine
  - A container runtime (like Docker) > pulling container image from registry(where image is saved), unpacking contianer and run the application.

# Troubleshooting with kubectl

- kubectl get > list resources
- kubectl describe > detailed information about a resource.
- kubectl logs > print logs
- kubectl exec > execute a command

## Command line examples

- Previously we have created an applicaiton. At first see its running or not. `kubectl get pods`.
- If we want to see the detailed information about our application. we can run `kubectl describe pods`.
- The pods are isolated, to communicate with them we will create proxy to other terminal. `kubectl proxy`.
- Now we will get the pod name and query with the pod name. We can use enviroment variable for that.
- `export POD_NAME=$(kubectl get pods -o go-template --template '{{range.items}}{{.metadata.name}}{{"\n"}}{{end}})`
- We can check the variable. `echo $POD_NAME`
- lets check it with curl. `curl http://localhost:8001/api/v1/namespace/default/pods/$POD_NAME`
- We can check the logs. We have only one container, therfore we don't need to tell container name. `kubectl logs $POD_NAME`
- We can execute command inside container.
- `kubectl exec $POD_NAME -- env` will show environemnt variables inside the container.
- `kubectl exec -ti $POD_NAME -- bash` will create a bash session. So we have a console inside our container.
- We can see any file. Like our applicaton source file inside server.js file. We can see it `cat server.js`
- We can use `curl localhost:8080`
- We can `exit` to close the container

# Using a **service** to expose the app

- Service is a abstraction, which is defined by `YAML(preferred)` filetype or `json` filetype.
- The sets of pods is usually defined by _LabelSelector_, there are some you want also without a label.
- There are some type of

  - ClusterIP (default) - Exposes the Service on an internal IP in the cluster. This type makes the Service only reachable from within the cluster.
  - NodePort - Exposes the Service on the same port of each selected Node in the cluster using NAT. Makes a Service accessible from outside the cluster using <NodeIP>:<NodePort>. Superset of ClusterIP.
  - LoadBalancer - Creates an external load balancer in the current cloud (if supported) and assigns a fixed, external IP to the Service. Superset of NodePort.
  - ExternalName - Maps the Service to the contents of the externalName field (e.g. foo.bar.example.com), by returning a CNAME record with its value. No proxying of any kind is set up. This type requires v1.7 or higher of kube-dns, or CoreDNS version 0.0.8 or higher.

- Services match with labels and selector. Labels are key/value pairs attached to objects and can be used in any number of ways:

  - Designate objects for development, test, and production
  - Embed version tags
  - Classify an object using tags

## Command line

- Before staring anything we need to make sure our app is running. We can see the pods. `kubectl get pods`
- We can see all available services. `Kubectl get services`
  - Normally a service named as kubernetes will be available. It is created when minikube starts the cluster.
- Minikube doesnot support `LoadBalancer` at present. So we will use _NodePort_
  - `kubectl expose deployment/kubernetes-bootcamp --type="NodePort" --port 8080`
  - We can check it by again running `kubectl get services`
- Now we will see that there will be another service running named as kubernetes-bootcamp
- To find which port it is exposed, we can try `kubectl describe services/kubernetes-bootcamp`
- Now we will put it in an environment variable.
  - `export NODE_PORT=$(kubectl get services/kubernetes-bootcamp -o go-template='{{(index.spec.ports. 0).nodePort}}')`
- We can check our app curl $(minikube ip):$NODE_PORT
- We will see our app is running from the response of curl.
- Lets see the label. If we run the command to see the deployment. `kubectl describe deployment`
- The Deployment created automatically a label. At present our label name is app=kubernetes-bootcamp
- We can use this label to query avialable pods. `kubectl get pods -l app=kubernetes-bootcamp`, here l stands for label
- We can also see available services. `kubectl get services -l app=kubernetes-bootcamp`
- Create again environment variable `export POD_NAME=$(kubectl get pods -o -go-template --template '{{range.items}}{{.metadata.name}}{{"\n"}}{{end}}')`
- To apply new label. we need `kubectl label pods $POD_NAME version=v1`

  - this will apply new label to our pods, we also pinned the version
  - we can query now list of pods using new label. `kubectl get pods -l version=v1`

- To delete a service we can use `kubectl delete service -l app=kubernetes-bootcamp`
- We can check it `kubectl get services`
- To confirm this route is not exposed, we can use `curl $(minikub ip):$NODE_PORT` we will see failure
- So our pod is not running outside cluster. Lets see whether it is running inside the cluster.
  - `kubectl exec -ti $POD_NAME --curl localhost:8080` We will be seeing the app is running inside cluster.

# Running Multiple Instances of our app (Scaling)

- Scaling out a Deployments ensure that new pods are created and scheduled to Nodes with available resources.
- autoscaling is also available in kubernetes. But it will not be described here.

## Command line

- We will see first our available deployments using `kubectl get deployments`
  - we can see the output `NAME READY UP-TO-DATE AVAILABLE AGE kubernetes-bootcamp 1/1 1 1 11m`
- To see ReplicaSet created we need to run `kubectl get rs`
  - The output will be [Deployment-name]-[Random-number]
  - Two important columns will be _DESIRED_ and _CURRENT_
- Now scale the app make 4 replicas `kubectl scale deployments/kubernetes-bootcamp --replicas=4`
- Again see the available deployments. `kubectl get deployments`
- Now we will see the available and ready number 4 is there
- Let's see number of pods changed or not. `kubectl get pods -o wide`
- The change was registered in Deployments events log. we can check `kubectl describe deployments/kubernetes-bootcamp`

### Step 2: Load balancing

- To see available ports, we can use `kubectl describe services/kubernetes-bootcamp`
- As earlier create an environment variable. `export NODE_PORT=$(kubectl get services/kubernetes-bootcamp -o go-template='{{(index.spec.ports 0).nodePort}}')`
- Now using curl lets check it `curl $(minikube ip):$NODE_PORT`

### Step 3: Scale down

- `kubectl scale deployments/kubernetes-bootcamp --replicas=2`
- `kubectl get deployments` will show available services.
- `kubectl get pods -o wide` will show available running pods. others will show terminated.

# Performing a **rolling update**

- Rolling updates allows users to the following actions:
  - Promote applications from one environment to another(via container immage updates).
  - Rollback to previous options.
  - Continuous Integration and Continous Delivery of application with zero downtime.

## Command line

### Step 1: Update the version of the app

- At first we can see our deployments
  - `kubectl get deployments`
- Then get available pods can be seen
  - `kubectl get pods`
- To see current Image version of the app, we can see describe pods **Image** field
  - `kubectl describe pods`
- To update the application image we are telling to use the new version of the image with `set image` option
  - `kubectl set image deployments/kubernetes-bootcamp=jocatalian/kubernetes-bootcamp:v2`
- Now you will see updated pods `kubectl get pods`

### Step 2: Verify an update

- First checking the app is running and exposed port number can be seen by running
  - `kubectl describe services/kubernetes-bootcamp`
- Create an environment variable called NODE_PORT
  - `export NODE_PORT=$(kubectl get services/kubernetes-bootcamp -o go templates='{{(index.spec.ports 0).nodePort}}')`
- Try curl to check it
  - `curl $(minikube ip):$NODE_PORT`
- We can see the image number is always v2
- We can confirm the roll out by seeing rollout status
  - `kubectl rollout status deployment/kubernetes-bootcamp`
- `kubectl describe pods` **Image** field can also confirm that, which version is running.

### Step 3: Rollback to an update

- Let's perform another update and we set it as v10
  - `kubectl set image deployments/kubernetes-bootcamp kubernetes-bootcampl=gcr.io/google-samples/kubernetes-bootcamp:v10`
- Let's see available deployments.
  - `kubectl get deployments`
- We will not see desired amount of deployment `kubectl get deployments`
- let's check available pods `kubectl get pods`
  - We will see some pods has status **ImagePullBackoff**
- To get more insight of the problem `kubectl describe pods`. In the event section we will see the iamge is not available in the directory.
- To rollback to old working version we can use `rollout undo`
  - kubectl rollout undo deployments/kubernetes-bootcamp
- To check the pods run `kubectl get pods`
- For detailed reports see `kubectl describe pods`. we will see v2 is running
