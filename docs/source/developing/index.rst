.. _docker-stuff:

Docker
==================================
One very useful tool to have is a local build environment of the pyKLIP package to work on. We will be using a software container platform called Docker and this tutorial will provide a brief overview on what it is, how to set it up, and how to use it with respect to pyKLIP.  

Why use Docker/Local Build Environments?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependency Isolation
^^^^^^^^^^^^^^^^^^^^
One of the most important features a local build environment provides. The ability to isolate dependences is an invaluable tool in the development and deployment of any application. Docker creates images that wrap source code, configuration files, packages, services, etc for executing applications. Using one allows you to check if your code will build without failure and can be used as a sandbox to test your new features.

Images
^^^^^^
Another benefit is the use of images, specifically the vast collection of pre-built images on `Docker Hub <https://hub.docker.com/>`__. One example of this is the anaconda3 image we use to build the image for the pyKLIP pipeline. Images can be created off of and build on top of other existing images. 

One Image, One Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^
Using a docker image also provides a single, reproducible environment that everyone on the team can work on. This eliminates much of the hassle of the set up and preparation phase to work on a package. With an image, everyone works on the same, reproducible environement and gets rid of the whole "But it worked on my machine!!" issue. If it works on your image, it'll work on everyone else's image.

Multiple Build Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you get docker running, you can build multiple images! Want just an image with a clean install of Ubuntu? `Use the Ubuntu image <https://hub.docker.com/_/ubuntu/>`__! Want one image with `python 2 <https://hub.docker.com/r/continuumio/anaconda/>`__ and another that runs python 3? What about python `3.4 <https://hub.docker.com/r/continuumio/anaconda3/builds/bae8pwnyl7fqhxk2zvkan4c/>`__ vs `3.6 <https://hub.docker.com/r/continuumio/anaconda3/>`__ ? Create an image! (Almost) Anything is possible! After you've created the image, you can then upload it and share it with anyone via Docker Hub, so anyone on your team can use the same image as well. 


.. toctree::
   :maxdepth: 2
   :caption: Contents

   setup
   using
   sharing

