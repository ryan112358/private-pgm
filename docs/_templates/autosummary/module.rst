{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree: .
      :nosignatures:

      {% for item in functions %}
         {{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree: .
      :nosignatures:

      {% for item in classes %}
         {{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree: .
      :nosignatures:

      {% for item in exceptions %}
         {{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}
