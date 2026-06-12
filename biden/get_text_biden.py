"""
Cassady Shoaff | Montclair State University | APLN580 Corpus Linguistics | Spring 2021
Webscraper to collect raw text for corpus of Presidential Speeches & Remarks
"""
import requests
from bs4 import BeautifulSoup
from time import sleep
from random import randint

## Data Collection
# go to each link and gather the raw text content
counter = 0 # count links 
with open('urls.txt') as urls:
    for url in urls: # access each link
        # save each speech to file: president_speeches_000.txt
        filename = f"biden_speeches_{counter:03}.txt"
        file = open(filename, "w") 
        page = requests.get(url.strip()) # must strip newline char from end of url
        soup = BeautifulSoup(page.text,'html.parser') # create soup object of whole page
        # add title and date header tags
        title = soup.find('h1').get_text().strip()
        date = soup.find('time').get_text()
        file.writelines(f'<title="{title}">'+"\n")
        file.writelines(f'<date="{date}">'+"\n")
        # helper print statement
        print("Page",counter,url,title,date)
        # collect main speech content
        for br in soup.find_all('br'): br.replace_with("\n" + br.text) # <br><br> convert \n
        for p in soup.find_all('p'): # find all paragraphs <p>
            line = p.get_text() # get text (exclude HTML tags)
            file.writelines(" "+line+"\n") # write each paragraph to file, add whitespace
        sleep(randint(2,10)) # limit requests
        file.close()
        counter += 1 # increment link counter
        # if counter > 3: break # toggle comment if you need to test just a few pages
    print(f"Successfully collected {counter} pages")
urls.close()
