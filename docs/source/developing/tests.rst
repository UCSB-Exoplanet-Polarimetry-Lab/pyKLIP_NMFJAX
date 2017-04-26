.. _tests-label:


#######
Testing
#######

Here we will go over how we test and what our testing infrastructure looks like at pyKLIP.

All of our tests can be found in the ``tests`` directory. In the directory, each module or feature gets it's own test
file, and inside every file each function is a different tests for the module/feature.

The testing workflow for pyKLIP can be broken down into the following steps:

* Creating the tests
* Documenting the tests
* Running the tests


Creating Tests
==============
All tests for pyKLIP can be found in the "tests" directory. We use pytest to run all of our tests in this directory. All
 tests should be named "test_<module/purpose>", and within the test files each function should be named "test_<function
 name>" to give an idea of what the test is for. The docstring for the function will go into detail as to what the test
 is testing and a summary of how it works.

Our testing framework is organized so that each file tests an individual module or feature, and each function inside
each test file tests different aspects of the module/feature.

All pathing should be absolute paths.

Some commands you may find helpful:

* **os.path.abspath(path)** - returns the absolute path of the path provided
* **os.path.dirname(path)** - returns the name of the directory of the path provided.
* **os.path.exists(path)** - returns True if the path exists, False otherwise.
* **os.path.sep** = path separator. This is important because different os can have different path separators. For example ubuntu linux uses "/" while windows uses "\\". This will take care of that.
* **os.path.join(args)** - returns a string with all the args separated by the appropriate path separator. ex.)os.path.join("this", "is", "a", "path") would return "this/is/a/path" in ubuntu linux.
* **__file__** - pathname of file this is written in.
* **sys.path.sep** - the appropriate path separator for each os.


Documenting Tests
=================
Docstring for tests should follow this format::

    """
    Summary of what your tests does goes here.

    Args:
        param1: First param.
        param2: Second param.
        etc: etc

    Returns:
        A description of what is returned.

    Raises:
        Error: Exception.
    """

Use the `following link <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`__ for more info on
docstrings as well as python style in general.

Running Tests
=============
All of our tests are run using pytest on the test directory on a docker anaconda image of our bitbucket pipeline. If
these terms seem unfamiliar, please refer to our :ref:`developing-label` page under the "Docker" section for more
information on docker and our pipeline.

Here is a simple overview of the steps involved in running the tests in our pipeline:

1. Bitbucket Pipelines reads our pipeline yml file to build the pipeline.
2. Creates a docker image of the latest continuum anaconda3.
3. Git clones the pyklip repository inside image.
4. Installs all necessary packages.
5. Runs tests using pytest on the test directory.
6. Runs coverage analysis on our tests.
7. Submits coverage report.

To simply run a single test you can either call the file directly using::

    $ python <Test file name>.py

Otherwise we can also use pytest for more flexibility::

    $ python -m pytest [args]
    $ pytest [args]

The above line will invoke pytest through the Python interpreter and add the current directory to sys.path. Otherwise
the second line is equivalent to the first.
There are many arguments and many different ways to use pytest. To run a single test simply enter the path to the test
file to run, to test all files in a directly use the path to the directory instead of a single file.
For more information on how to use pytest and some of its various usages, visit `this link <https://docs.pytest.org/en/latest/usage.html#>`__.
