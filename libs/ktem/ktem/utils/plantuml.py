#!/usr/bin/env python

from __future__ import print_function

import base64
import string
from zlib import compress
import re
import logging

import httpx
import six  # type: ignore

if six.PY2:
    from string import maketrans
else:
    maketrans = bytes.maketrans

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

plantuml_alphabet = (
    string.digits + string.ascii_uppercase + string.ascii_lowercase + "-_"
)
base64_alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"
b64_to_plantuml = maketrans(
    base64_alphabet.encode("utf-8"), plantuml_alphabet.encode("utf-8")
)


class PlantUMLError(Exception):
    """
    Error in processing.
    """


class PlantUMLConnectionError(PlantUMLError):
    """
    Error connecting or talking to PlantUML Server.
    """


class PlantUMLHTTPError(Exception):
    def __init__(self, response, content):
        self.response = response
        self.content = content
        message = "%d: %s" % (self.response.status_code, self.response.reason_phrase)
        super().__init__(message)


def deflate_and_encode(plantuml_text):
    """zlib compress the plantuml text and encode it for the plantuml server."""
    zlibbed_str = compress(plantuml_text.encode("utf-8"))
    compressed_string = zlibbed_str[2:-4]
    return (
        base64.b64encode(compressed_string).translate(b64_to_plantuml).decode("utf-8")
    )


class PlantUML:
    def __init__(self, url: str = "http://www.plantuml.com/plantuml/svg/"):
        self.url = url
        self.client = httpx.Client()

    def process(self, plantuml_text: str) -> str:
        """
        Processes the PlantUML text into an SVG image.
        
        Args:
            plantuml_text (str): The PlantUML code.
        
        Returns:
            str: The SVG content.
        
        Raises:
            PlantUMLHTTPError: If the PlantUML server returns an error.
        """
        # Remove any leading/trailing whitespace
        plantuml_text = plantuml_text.strip()

        # Remove code fences if present
        plantuml_text = re.sub(r'^```plantuml\s*', '', plantuml_text, flags=re.MULTILINE)
        plantuml_text = re.sub(r'```$', '', plantuml_text, flags=re.MULTILINE)

        # Encode PlantUML text
        encoded_text = deflate_and_encode(plantuml_text)

        # Construct the final URL
        final_url = self.url + encoded_text

        try:
            response = self.client.get(final_url)
            logger.info(f"PlantUML server responded with status code {response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"An error occurred while requesting {e.request.url!r}: {e}")
            raise PlantUMLConnectionError(f"An error occurred while requesting {e.request.url!r}.") from e

        if response.status_code != 200:
            raise PlantUMLHTTPError(response, response.content)

        svg_content = response.text
        svg_content = svg_content.replace("<svg ", "<svg id='mindmap' ")

        # Wrap in fixed height div
        svg_content = (
            "<div id='mindmap-wrapper' "
            "style='height: 400px; overflow: hidden;'>"
            f"{svg_content}</div>"
        )

        return svg_content
