import os
import sys
from nltk.tokenize import RegexpTokenizer
from nltk import clean_html
import urllib.request
from bs4 import BeautifulSoup
import sys
import re
import urllib.parse
import hashlib
import string
import codecs
from nltk.stem import PorterStemmer
import socket
import nltk
from nltk.corpus import wordnet
from collections import Counter
import pandas as pd
import numpy
import csv
import operator
import requests
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from urllib.request import urlopen


class Crawler:
    def __init__(self, seed_url):
        self.seed_url = seed_url
        self.domain_url = "/".join(self.seed_url.split("/")[:3])
        self.robots_txt = None
        self.stop_words_file = None
        self.page_limit = 60
        self.docValue = []
        self.stop_words = []  # list of words to be ignored when processing documents
        self.url_frontier = []  # list of urls not yet visited
        self.visited_urls = {}  # URL : (Title, DocumentID) (hash of content of a visited URL)
        self.outgoing_urls = []
        self.broken_urls = []
        self.graphic_urls = []
        self.filtered_sentence = []
        self.htmlUrl = []
        self.wordList = []
        self.ngrams2 = {}
        self.keyWord = []
        self.docKeyWord = []
        self.TermList = []
        self.keywordFrequency = {}
        self.terms = {}
        self.all_terms = []  # set of all stemmed terms in all documents
        self.frequency_matrix = []
        self.frequency_matrixColumn = []  # Term doc frequency matrix (row=term, col=doc)
        self.num_pages_crawled = 0  # number of valid pages visited
        self.num_pages_indexed = 0  # number of pages whose words have been stored

        """
        note: the attributes below only contain information from those documents whose 
              words were saved (.txt, .htm, .html, .php)
        """
        self.duplicate_urls = {}  # DocumentID : [URLs that produce that ID]
        self.doc_urls = {}  # DocumentID: first URL that produces that ID
        self.doc_titles = {}  # DocumentID : title
        self.doc_words = {}  # DocumentID : [words]

        # print the report produced from crawling a site

    def __str__(self):
        report = "\nPages crawled: " + str(self.num_pages_crawled) \
                 + "\nPages indexed: " + str(self.num_pages_indexed) \
                 + "\nVisited URLs: " + str(len(self.visited_urls)) \
                 + "\nVisited Urls: " + "\n  +  " + "\n  +  ".join(self.visited_urls) \
                 + "\nDoc Urls: " + "\n  +  " + "\n  +  ".join(self.doc_urls) \
                 + "\nhtml/php/txt Urls: " + str(len(self.htmlUrl)) \
                 + "\n\nText URLs: " + "\n  +  " + "\n  +  ".join(self.htmlUrl) \
                 + "\n\nOutgoing URLs: " + "\n  +  " + "\n  +  ".join(self.outgoing_urls) \
                 + "\n\nBroken URLs: " + "\n  +  " + "\n  +  ".join(self.broken_urls) \
                 + "\n\nGraphic URLs: " + "\n  +  " + "\n  +  ".join(self.graphic_urls) \
                 + "\n\nDuplicate URLs:\n"

        # print duplicate urls
        for key in range(len(self.duplicate_urls.keys())):
            report += "\t +  Doc" + str(key + 1) + ":\n"
            for val in list(self.duplicate_urls.values())[key]:
                report += "\t\t  +  " + val + "\n"

        return report

    '''
    Returns a dictionary of allowed and disallowed urls
    Adapted from https://stackoverflow.com/a/43086135/8853372
    '''

    def get_robots_txt(self):
        # open seed url
        result = urllib.request.urlopen(self.seed_url + "/robots.txt").read()
        result_data_set = {"Disallowed": [], "Allowed": []}

        # for reach line in the file
        for line in result.decode("utf-8").split('\n'):
            if line.startswith("Allow"):
                result_data_set["Allowed"].append(
                    self.seed_url + line.split(": ")[1].split('\r')[0]
                )  # to neglect the comments or other junk info
            elif line.startswith("Disallow"):  # this is for disallowed url
                result_data_set["Disallowed"].append(
                    self.seed_url + line.split(": ")[1].split('\r')[0]
                )  # to neglect the comments or other junk info

        return result_data_set

    # sets the stop words list given a file with stop words separated by line
    def set_stop_words(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as stop_words_file:
                stop_words = stop_words_file.readlines()

            self.stop_words = [x.strip() for x in stop_words]
            self.stop_words_file = filepath

        except IOError as e:
            print("Error opening" + filepath + " error({0}): {1}".format(e.errno, e.strerror))
        except ValueError:
            print("Error opening" + filepath + ": Data is not correctly formatted. See README.")
        except:
            print("Error opening" + filepath + "Unexpected error:", sys.exc_info()[0])
            raise

    '''
    returns whether or not a url is valid
    source: https://stackoverflow.com/a/7160778/8853372
    '''

    def url_is_valid(self, url_string):
        pattern = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$',
            re.IGNORECASE)

        return bool(pattern.match(url_string))

    '''
    returns whether or not a given word is valid
    A word is a string of non-space characters, beginning with an alphabetic character. 
    It may contain special characters, but the last character of a word is either alphabetic or numeric.
    '''

    def word_is_valid(self, word):
        pattern = re.compile(r'^[a-zA-z](\S*)[a-zA-z0-9]$')

        return bool(pattern.match(word))

    # returns whether or not the url is within the scope of the seed url
    def url_is_within_scope(self, url_string):
        return self.seed_url in url_string

    '''
    produces a list of duplicate documents
    populates self.duplicate_urls with DocumentID : [URLs that produce that ID]
    '''

    def SearchFind(self):
        synonyms = []
        Term = []

        outputString = ""
        for j in iter(self.ngrams2):
            Term.append(j)

        for i in range(len(Term)):
            for syn in wordnet.synsets(Term[i]):
                for l in syn.lemmas():
                    self.TermList.append(l.name())
                    synonyms.append(l.name())

                    # print(self.all_terms[i])
        print(set(synonyms))

    def produce_duplicates(self):
        duplicates = {}

        # populate duplicates with DocumentID : [URLs]
        for url, (_, docID) in self.visited_urls.items():
            duplicates[docID] = [url] if duplicates.get(docID) is None else duplicates[docID] + [url]

        # set duplicate_urls to those instances that have only one URL
        self.duplicate_urls = {docID: urls for docID, urls in duplicates.items() if len(urls) > 1}

    # crawls a site and returns a dictionary of information found
    def crawl(self):
        self.robots_txt = self.get_robots_txt()
        self.set_stop_words('static/Input/stopwords.txt')

        print("robots.txt: " + " ".join("{}{}".format(key, [
            v.replace(self.domain_url, "") for v in val]) for key, val in self.robots_txt.items()) + "\n")

        self.url_frontier.append(self.seed_url + "/")

        '''
        pop from the URL frontier while the queue is not empty
        links in queue are valid, full urls
        '''
        while self.url_frontier and (self.page_limit is None or self.num_pages_indexed < self.page_limit):
            # current_page refers to the url of the current page being processed
            current_page = self.url_frontier.pop(0)  # select the next url

            # calculate present working directory
            pwd = "/".join(current_page.split("/")[:-1]) + "/"

            if pwd not in self.robots_txt["Disallowed"]:
                try:

                    # hit the current page
                    handle = urllib.request.urlopen(current_page)
                    # current_page=self.cleanUrl(current_page)

                # basic HTTP error e.g. 404, 501, etc
                except urllib.error.HTTPError as e:
                    if current_page not in self.broken_urls and current_page is not None:
                        self.broken_urls.append(current_page)

                else:
                    current_content = str(handle.read())

                    # convert content to BeautifulSoup for easy html parsing
                    soup = BeautifulSoup(current_content, "lxml")

                    # grab the title of the page, store file name if title isn't available (e.g. PDF file)
                    current_title = str(soup.title.string) if soup.title is not None else current_page.replace(pwd, '')

                    # hash the content of the page to produce a unique DocumentID
                    current_doc_id = hashlib.sha256(current_content.encode("utf-8")).hexdigest()

                    # mark that the page has been visited by adding to visited_url
                    self.visited_urls[current_page] = (current_title, current_doc_id)
                    self.num_pages_crawled += 1

                    print(str(self.num_pages_crawled) + ". " + "Visiting: " +
                          current_page.replace(self.domain_url, "") + " (" + current_title + ")")

                    # if the page is an html document, we need to parse it for links
                    if any((current_page.lower().endswith(ext) for ext in ["/", ".html", ".htm", ".php", ".txt"])):
                        self.htmlUrl.append(current_page)

                        [s.extract() for s in soup('title')]

                        # format the content of the page
                        formatted_content = codecs.escape_decode(bytes(soup.get_text().lower(), "utf-8"))[0].decode(
                            "utf-8", errors='replace')

                        # store only the words of the file
                        content_words = list(re.sub(r'https?:\/\/\S*', '', formatted_content).split())
                        content_words = list(re.sub('[' + string.punctuation + ']', '', formatted_content).split())

                        # remove the 'b' character that's prepended when converting
                        content_words[0] = content_words[0][1:]

                        # keep track of only those words that are valid and not in the stop word collection
                        self.doc_words[current_doc_id] = [w for w in content_words
                                                          if w not in self.stop_words and self.word_is_valid(w)]

                        # store the title
                        self.doc_titles[current_doc_id] = current_title

                        # store the url if it hasn't been stored already (to avoid duplicates)
                        if current_doc_id not in self.doc_urls:
                            self.doc_urls[current_doc_id] = current_page

                        self.num_pages_indexed += 1

                        # go through each link in the page
                        for link in soup.find_all('a'):
                            # current_url refers to the current link within the current page being processed
                            current_url = link.get('href')

                            # expand the url to include the domain
                            if current_url is not None and pwd not in current_url:
                                # only works if the resulting link is valid
                                current_url = urllib.parse.urljoin(pwd, current_url)

                            # the link should be visited
                            if current_url is not None and self.url_is_valid(current_url):

                                # the link is within scope and hasn't been added to the queue
                                if self.url_is_within_scope(current_url) and current_url not in self.url_frontier:

                                    # ensure the hasn't been visited before adding it to the queue
                                    if current_url not in self.visited_urls.keys():
                                        self.url_frontier.append(current_url)

                                elif not self.url_is_within_scope(
                                        current_url) and current_url not in self.outgoing_urls:
                                    self.outgoing_urls.append(current_url)

                            # the link is broken
                            elif current_url not in self.broken_urls and current_url is not None:
                                self.broken_urls.append(current_url)

                    # file is a graphic, mark it as such
                    elif any(current_page.lower().endswith(ext) for ext in [".gif", ".png", ".jpeg", ".jpg"]):
                        self.graphic_urls.append(current_page)

            else:
                print("Not allowed: " + current_page.replace(self.domain_url, ""))

        # dictionary containing information about the site

    def cleanInput(self, input):
        input = re.sub('\n+', " ", input).lower()
        input = re.sub('\t', " ", input).lower()
        input = re.sub('\r', " ", input).lower()
        input = re.sub('\[[0-9]*\]', "", input)
        input = re.sub(' +', " ", input)
        # input=re.compile(r'@<[^>]+>\s+(?=<)|<[^>]+>').sub(" ",input)
        input = re.sub(r'@<[^>]+>\s+(?=<)|<[^>]+>', " ", input)
        input = re.sub(r'https?:\/\/\S*', " ", input)
        input = re.sub(r'&nbsp', " ", input)
        input = re.sub(r' \d+', " ", input)

        input = bytes(input, "UTF-8")
        input = input.decode("ascii", "ignore")
        cleanInput = []
        input = input.split(' ')
        for item in input:
            item = item.strip(string.punctuation)
            if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
                cleanInput.append(item)
        return cleanInput

    def ngrams(self, input, n):
        input = self.cleanInput(input)

        output = {}

        for i in range(len(input) - n + 1):
            ngramTemp = " ".join(input[i:i + n])
            if ngramTemp not in output:
                output[ngramTemp] = 0
            output[ngramTemp] += 1

        self.ngrams2 = output.copy()
        """for i in iter(output):
            if i not in self.stop_words:
                self.terms = output.copy()"""

        return output

    def wordCountWrite(self):
        count = 1
        for i in range(len(self.htmlUrl)):
            print(self.htmlUrl[i])
            content = str(urlopen(self.htmlUrl[i]).read(), 'utf-8')
            ngrams = self.ngrams(content, 1)
            sortedNGrams = sorted(ngrams.items(), key=operator.itemgetter(1), reverse=True)

            return sortedNGrams

    def keywordWrite(self):
        content2 = str(urlopen(self.seed_url).read(), 'utf-8')
        ngramsKey = self.ngramsKeyWord(content2, 1)
        sortedNGrams2 = sorted(ngramsKey.items(), key=operator.itemgetter(1), reverse=True)
        count2 = 0
        for x, y in iter(sortedNGrams2):
            if x not in self.stop_words:
                count2 += 1
                if count2 == 6:
                    break
                    # print(x)
                self.keyWord.append((x, y))

    def docKeyWordWrite(self):
        count3 = 0
        for i in range(len(self.htmlUrl)):
            print(self.htmlUrl[i])
            content3 = str(urlopen(self.htmlUrl[i]).read(), 'utf-8')
            ngrams3 = self.ngramsKeyWord(content3, 1)
            sortedNGrams3 = sorted(ngrams3.items(), key=operator.itemgetter(1), reverse=True)
            for x, y in iter(sortedNGrams3):
                if x in self.keyWord:
                    self.docKeyWord.append((x, y))

    def ngramsKeyWord(self, input, n):
        input = self.cleanInput(input)
        output = {}

        for i in range(len(input) - n + 1):
            ngramTemp = " ".join(input[i:i + n])
            if ngramTemp not in output:
                output[ngramTemp] = 0
            if ngramTemp in output and ngramTemp not in self.stop_words:
                output[ngramTemp] += 1

        self.keywordFrequency = output.copy

        return output
