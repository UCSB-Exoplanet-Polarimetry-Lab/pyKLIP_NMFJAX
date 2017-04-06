.. _developing-label:

Developing for pyKLIP
============================

Overview
--------
This is a guide for developing for the pyKLIP package. 
In this section we will go over:

1. Our Build
2. Docker
3. Code Coverage and Testing


1. Our Build
-------------
Most of the code for pyKLIP is written in python, specifically python versions 2.7 and 3.6. It is to be noted, however, that as we are using the `anaconda3 image <https://hub.docker.com/r/continuumio/anaconda3>`__ for our build which uses python version 3.6, so make sure all code can run in python 3. Anaconda comes with many useful packages built-in so if you need a package check out `the following link <https://docs.continuum.io/anaconda/pkg-docs>`__ before installing anything and see if it's already included. 



2. Docker
---------
Refer to :ref:`docker-stuff` for more information on Docker. 

One very useful tool to have is a local build environment of the pyKLIP package to work on. We will be using a software container platform called Docker and this tutorial will provide a brief overview on how to set it up and how to use it with respect to pyKLIP. 

All of the following information and more can be found `here <https://docs.docker.com/engine/getstarted/>`__. This only serves as a summary for developing for pyKLIP. 

Installation
^^^^^^^^^^^^
We will be using the community edition of Docker.

For Windows and Mac, installation instructions and requirements can be found `here <https://docs.docker.com/engine/getstarted/step_one/>`__. 

For Linux ::

        sudo apt-get update
        sudo apt-get install docker-ce


Using Docker
^^^^^^^^^^^^
For a very basic tutorial on Docker and how to use it, check out the docker docs and their tutorials `here <https://docs.docker.com/engine/getstarted/step_three/#step-2-run-the-whalesay-image>`__. There are a lot of helpful tutorials and information there. 

From a fresh install, there are a few steps to getting your container up and running. 

1. Download and run the pyKLIP image. You can do this by pulling and running, but simply running the pyKLIP image will do both steps in one. Executing the run command will first check your local machine for the appropriate images and use them if Docker finds them, or download them from Docker Hub if it fails. ::

        $ docker run simonko/pyklip
2. From here, to check if the appropriate image has been set up use the :mod:`docker images` command, and you should get something similar to the following ::

        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        simonko/pyklip          latest              e9a584c685bb        4 hours ago         2.37 GB
3. Running the command below creates a container of the pyklip image and gives us an interactive shell to interact with container. The ``-i -t`` flags allows for interactive mode and allocates a pseudo-tty for the container respectively. This is usually combined into the flag ``-it``. If you don't specify a tag, it'll generate some random name for you. (ex. sad_lovelace, agitated_saha, ecstatic_pare, etc) ::

        $ docker run -it simonko/pyklip:latest /bin/bash
4. When you're done with the container, simply type ``exit`` and your session will end. If you get the message that states there is a process running, simply type exit again and it'll exit the session. 
5. After you've made your container you should be able to see it with ::
        
        $ docker ps -a

        CONTAINER ID        IMAGE                   COMMAND                  CREATED             STATUS                     PORTS               NAMES
        c6695e4d9a63        simonko/pyklip:latest   "/usr/bin/tini -- ..."   6 seconds ago       Exited (0) 3 seconds ago                       zealous_goldwasser
6. To get into the container, you have to first start the container again, then use the attach command to get back into the interactive shell. ::

        $ docker start <container name>
        $ docker attach <container name>

Using Local Files
^^^^^^^^^^^^^^^^^
Once you have your image, you can cp over local files into the container. To do this you have to use the ``attach`` command and ``-d`` flag like so ::

        $ docker run -it -d simonko/pyklip:latest 

        exit

        $ docker cp <source file/directory> <container name>:<destination>

        $ docker start <container name>

        $ docker attach <container name>

It should be noted that if the specified destination does not exist, it will create the destination for you. For example if I were to do the following ::
        
        $ docker cp <somefile/directory> zealous_goldwasser:/pyklip
inside the `zealous_goldwasser` container and it did not already have a pyklip directory, docker would create the directory for me and place the file in it, just like the normal cp command. 

Deleting Images and Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You may find that your docker is getting a bit cluttered after playing around with it. The following section will show you how to delete images and containers. You can also refer to `this cheat sheet <https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes#a-docker-cheat-sheet>`__ for more on deleting images and containers. The below is just a few basic and useful commands. 

`Deleting Containers`
"""""""""""""""""""""

To delete a container, first locate the container(s) you wish to delete, then use ``docker rm <ID or NAME>`` to delete::

        $ docker ps -a

        CONTAINER ID        IMAGE                   COMMAND                  CREATED             STATUS                     PORTS               NAMES
        c6695e4d9a63        simonko/pyklip:latest   "/usr/bin/tini -- ..."   6 seconds ago       Exited (0) 3 seconds ago                       zealous_goldwasser

        $ docker rm <container ID (c6695e4d9a63) or Name (zealous goldwasser)>

To delete multiple containers at once use the filter flag. For example, if you want to delete all exited containers ::

        $ docker rm $(docker ps -a -f status=exited -q)
You can also find all containers all exited containers using just the command in the parenthesis without the -q flag. This is particularly useful if there are many exited containers and you don't remember which ones you wanted to delete. 

`Deleting Images`
"""""""""""""""""

To delete your images first you must find which ones you wish to delete. It should also be noted that to delete an image, there can be no containers attached to it. ::


        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        pyklip-pipeline         latest              e9a584c685bb        13 days ago         2.37 GB
        simonko/pyklip          latest              e9a584c685bb        13 days ago         2.37 GB
        localrepo               latest              dc74a96e5ef0        2 weeks ago         2.25 GB
        ubuntu                  latest              0ef2e08ed3fa        3 weeks ago         130 MB
        continuumio/anaconda3   latest              26043756c44f        6 weeks ago         2.23 GB

        $ docker rmi <repository name>

.. note::
        Before you delete an image, all containers using the image must be DELETED, not exited.

To delete ALL of your images ::

        $ docker rmi $(docker images -a -q)

Creating Images
^^^^^^^^^^^^^^^
In this section, you will learn how to create and upload your own image. To do this you need to make a dockerfile. If you wish to share the image for others to use, you need to create a Docker Hub account and push your image into a repository. This section will go over all of these steps. For a more detailed tutorial `use this link <https://docs.docker.com/engine/getstarted/step_four/#step-4-run-your-new-docker-whale>`__. Otherwise here are the very basics. 

Docker images are created from a set of commands in a dockerfile. What goes on this file is entirely up to you. Docker uses these commands to create an image, and it can be an entirely new one or an image based off of another existing image. 


1. Create a file and name it dockerfile. There are three basic commands that go on a dockerfile.
    - FROM <Repository>:<Build> - This command will tell docker that this image is based off of another image. You can specify which build to use. To use the most up-to-date version of the image, use "latest" for build. 
    - RUN <Command> - This will run commands in a new layer and creates a new image. Typically used for installing necessary packages. You can have multiple RUN statements.
    - CMD <Command> - This is the default command that will run once the image environment has been set up. You can only have ONE CMD statement. 
    For more information on RUN vs CMD here is a `useful link <http://goinbigdata.com/docker-run-vs-cmd-vs-entrypoint/>`__.
2. After you've made your file run the following command to create your image ::
    
        $ docker build -t <Image Name> <Path to Directory of Dockerfile>
The ``-t`` flag lets you name the image. 

For example, the docker file used for the pyklip image I set up above (under the "Using Docker" section) is made using a dockerfile with the following content: ::

        FROM continuumio/anaconda3:latest
        RUN git clone https://bitbucket.org/pyKLIP/pyklip.git \
         && pip install coveralls \
         && pip install emcee \
         && pip install corner \
         && conda install -c https://conda.anaconda.org/astropy photutils

Uploading Images
^^^^^^^^^^^^^^^^
1. If you haven't already, `create a Docker Hub account <https://hub.docker.com/register/?utm_source=getting_started_guide&utm_medium=embedded_MacOSX&utm_campaign=create_docker_hub_account>`__. 
2. After you've made your account, sign in and click on "Create Repository" and fill out the details. Make sure visibility is set to PUBLIC. Press create.
3. Find your image ID. Using a previous example ::

        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        pyklip-pipeline         latest              e9a584c685bb        13 days ago         2.37 GB

The image ID would be e9a584c685bb. 

4. Tag the image using ::
        
        $ docker tag <Image ID> <DockerHub Account Name>/<Image Name>:<Version or Tag>

So for the pyklip pipeline image my command would be: ::
        
        $ docker tag e9a584c685bb simonko/pyklip:latest 

Check that the image has been tagged ::

        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        pyklip-pipeline         latest              e9a584c685bb        13 days ago         2.37 GB
        simonko/pyklip          latest              e9a584c685bb        13 days ago         2.37 GB
5. Login to Docker on terminal ::
        
        $ docker login

        Username: *****
        Password: *****
        Login Succeeded
6. Push your tagged image to docker hub ::

        $ docker push <Repository Name> 

7. To pull from the repo now, all you have to do is run the repo. Docker will automatically pull from docker hub if it cannot find it locally. 



3. Coverage and Testing
-----------------------


