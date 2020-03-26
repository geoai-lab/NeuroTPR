# Automatic approach of constructing Geo-annotated dataset 

Here presents an automatic workflow of quickly constructing a geo-annotated dataset using wikipedia dump files

### Download Wikipedia dump file

Download the wikipedia dump files from [Wiki-dumpfile](https://dumps.wikimedia.org/enwiki/20200101/). Please use the dump file version as enwiki-20200101-pages-articles1.xml-p10p30302.bz2. Save the dump files into your directory

### Process dump file and extract location information

```bash
    python3 process_wikipages.py
 ```

 This script will process one wikipedia dump file, iterate on all wikipedia article, and do:

 * Extract the first paragraph of each wikipedia page, and save into a single txt file
 * Check the template box attribute of the each wikipedia article, save the place-related entities into DB

### Annotate the saved wikipedia-based texts

```bash
    python3 annotate_wikipages.py
 ```
This script will iterate the saved wikipedia texts, and annotate the link inside texts if the entities linked to it is a location
The output file is a WNUT-2017 format txt file

### Limitation
 The annoated dataset is not 100% precisely and clean-format in two perspectives:
 1. Not all places in wikipedia articles are in the form of hyperlink, which means a few places are not annotated;
 2. The embedded citation and image metadata description might be included, adding some data noise;

 It would be great to look through the generated dataset first before use it.