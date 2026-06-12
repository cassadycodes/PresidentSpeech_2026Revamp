"""
Cassady Shoaff | Montclair State University | APLN580 Corpus Linguistics | Spring 2021
Webscraper to collect raw text for corpus of Presidential Speeches & Remarks
"""
import requests
from bs4 import BeautifulSoup
from time import sleep
from random import randint

president = "trump"

## Data Collection
# go to each link and gather the raw text content
counter = 0 # count links 
with open('urls.txt') as urls:
    for url in urls: # access each link
        counter += 1 # increment link counter
        if counter < 15: # USE THIS IF CONNECTION INTERRUPTIONS PREVENT COLLECTION
            continue
        # save each speech to file president_speeches_000.txt
        file = open(f"{president}_speeches_{counter:03}.txt", "w", encoding="utf-8") 
        print("Page",counter,url) # helper print statement
        page = requests.get(url.strip()) # must strip newline char from end of url
        soup = BeautifulSoup(page.text,'html.parser') # create soup object
        # add title and date header tags
        title = soup.find('h1').get_text()
        date = soup.find('time').get_text()
        file.writelines(f'<title="{title}">')
        file.writelines(f'<date="{date}">')
        # collect main speech content
        # luckily <p> tags are only inside the main content section! very helpful :)
        for p in soup.find_all('p'): # find all paragraphs
            line = p.get_text() # get text (exclude HTML tags)
            file.writelines(line+" \n") # write each paragraph to file, add whitespace
        sleep(randint(2,10)) # limit requests
        file.close()
        # break # TOGGLE COMMENT TO TEST JUST 1 PAGE TO CHECK NEW CHANGES
    print(f"Successfully collected {counter} pages")
file.close()
