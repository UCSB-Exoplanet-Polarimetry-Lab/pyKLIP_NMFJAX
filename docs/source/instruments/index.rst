.. _instruments-label:

Instrument Tutorials
=====================

pyKLIP supports data from various instruments. Here we have compiled tutorials on how to use each specific insturment interface. 
pyKLIP also has a :py:class:`pyklip.instruments.Instrument.GenericData` class to support generic high-contrast imaging data when
your instrument doesn't yet have an interface and you don't yet want to make one. Note that GenericData works fine for standard
KLIP reductions, but may not work for the more sophisticated forward modelling libraries. 

.. toctree::
   :maxdepth: 1

   ../klip_gpi
   p1640
   sphere
   generic_data