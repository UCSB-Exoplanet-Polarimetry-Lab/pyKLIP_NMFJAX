.. _p1640-label:


Developing for pyKLIP
============================

Overview
--------
This is a guide for developing for the pyKLIP package. 
In this section we will go over:

1. Our Build
2. Setting up and using a local build environment with Docker
3. Code Coverage and Tests


Our Build
----------
Most of the code for pyKLIP is written in python, specifically python versions 2.7 and 3.6. It is to be noted, however, that as we are using the `anaconda3 image <https://hub.docker.com/r/continuumio/anaconda3>`__ for our build which uses python version 3.6, so make sure all code can run in python 3. Anaconda comes with many useful packages built-in so if you need a package check out `the following link <https://docs.continuum.io/anaconda/pkg-docs>`__ before installing anything and see if it's already included. 


Docker
------

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
        
        $ docker cp <somefile> zealous_goldwasser:/pyklip/
inside the zealous_goldwasser container and it did not already have a pyklip directory, docker would create the directory for me and place the file in it, just like the normal cp command. 