************
Contributing
************

We welcome contributions to `private-pgm`! Whether it's reporting a bug, proposing a new feature, improving documentation, or writing code, your help is appreciated.

Ways to Contribute
==================

* **Reporting Bugs:** If you find a bug, please open an issue on our `GitHub Issues <https://github.com/ryan112358/private-pgm/issues>`_ page. Include as much detail as possible: steps to reproduce, error messages, your environment (Python version, OS), and the `private-pgm` version.
* **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue to discuss it.
* **Documentation:** Improvements to the documentation are always welcome. If you find parts unclear or missing, feel free to make changes or suggest them.
* **Code Contributions:** If you'd like to contribute code, please follow the guidelines below.

Setting Up a Development Environment
====================================

1.  Fork the repository on GitHub.
2.  Clone your fork locally::

    git clone https://github.com/YOUR_USERNAME/private-pgm.git
    cd private-pgm

3.  Create and activate a virtual environment (see :doc:`installation`).
4.  Install dependencies and the package in editable mode::

    pip install -r requirements.txt
    pip install -e .[dev]  # Assuming you add a [dev] extra in setup.py for dev tools like pytest, flake8

Running Tests
=============
`private-pgm` uses `pytest` for testing. Tests are located in the ``test/`` directory.
To run the tests:

.. code-block:: bash

    pytest test/

Please ensure all tests pass before submitting a pull request. If you add new features, please include new tests.

Coding Style
============

* Please follow PEP 8 guidelines for Python code.
* Use clear and descriptive variable and function names.
* Add docstrings to new functions and classes (e.g., Google or NumPy style, compatible with Sphinx).
* (Optional: Add any specific linters or formatters you use, e.g., Flake8, Black)

Pull Request Process
====================

1.  Ensure your code adheres to the coding style and all tests pass.
2.  Create a new branch for your changes::

    git checkout -b feature/your-feature-name

3.  Commit your changes with clear and descriptive commit messages.
4.  Push your branch to your fork on GitHub::

    git push origin feature/your-feature-name

5.  Open a pull request from your fork to the main `private-pgm` repository.
6.  Clearly describe the changes in your pull request and link to any relevant issues.
7.  Be prepared to discuss your changes and make adjustments based on feedback.

Code of Conduct
===============
(Optional: If you have a Code of Conduct, link to it here.)

Thank you for contributing to `private-pgm`!
