import xml.sax
import mwparserfromhell


def process_article(text, infoboxes, template='Infobox'):
    """Process a wikipedia article looking for template"""

    # Create a parsing object
    wikicode = mwparserfromhell.parse(text)

    # Search through templates for the template
    matches = wikicode.filter_templates(matches=template)

    if len(matches) < 1:
        return False

    else:
        # Extract information from infobox
        for match in matches:
            # all_params = [param.name.strip_code().strip() for param in match.params if param.value.strip_code().strip()]
            # if "coordinates" in all_params:
            if str(match.name).rstrip() in infoboxes:
                print(match.name)
                return True

        return False


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self, infobox_set):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []
        self._infobox_set = infobox_set

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            infobox_page = process_article(self._values['text'], self._infobox_set, template='Infobox')

            if infobox_page:
                self._pages.append((self._values['title'], self._values['text']))
                print("add one article into the dataset!")
