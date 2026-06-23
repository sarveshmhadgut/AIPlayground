import pytest
from main import main
from io import StringIO
import sys

def test_main_function_output():
    # Capture the standard output
    captured_output = StringIO()
    sys.stdout = captured_output

    # Call the main function
    main()

    # Reset redirect.
    sys.stdout = sys.__stdout__

    # Assert that the output is as expected
    assert captured_output.getvalue().strip() == "Hello from langgraphsandbox!"
