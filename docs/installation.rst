************
Installation
************

This page guides you through installing the `private-pgm` library and its dependencies.

Prerequisites
=============

* Python (version 3.8 or higher recommended). You can check your Python version by running ``python --version``.
* `pip` (Python package installer).

Setting up a Virtual Environment (Recommended)
==============================================

It is highly recommended to use a virtual environment to manage project-specific dependencies. This avoids conflicts with other Python projects or your system's Python installation.

1.  Create a virtual environment (e.g., named ``.venv``)::

    python -m venv .venv

2.  Activate the virtual environment:

    * On macOS and Linux::

        source .venv/bin/activate

    * On Windows::

        .\.venv\Scripts\activate

Installation from Source
========================

Since `private-pgm` includes a `setup.py` file, you can install it directly from your local clone of the repository.

1.  **Clone the repository (if you haven't already):**

    .. code-block:: bash

        git clone https://github.com/ryan112358/private-pgm.git
        cd private-pgm

2.  **Install the package and its dependencies:**

    The project includes a `requirements.txt` file that lists the necessary dependencies.

    Navigate to the root directory of the project (where `setup.py` and `requirements.txt` are located) and run:

    .. code-block:: bash

        pip install .

    Alternatively, for development, you can install it in editable mode. This allows you to make changes to the source code and have them reflected immediately without reinstalling:

    .. code-block:: bash

        pip install -e .


Verify Installation
===================

After installation, you can verify it by opening a Python interpreter and trying to import the library:

.. code-block:: bash

    pytest test/*.py

If you encounter any issues, please refer to the project's issue tracker on GitHub.
